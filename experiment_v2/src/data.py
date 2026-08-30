"""Leakage-safe loading of the RF spectral-scan data.

The raw corpus contains one CSV for every
``(jammer channel, physical condition, scan id)`` tuple.  A scan id is the
experimental unit used for splitting: files are assigned to a split *before*
any overlapping windows are made.  Consequently, adjacent 32-sample windows
may overlap within a scan, but no source scan can occur in two splits.

Only NumPy and pandas are required.  The returned objects deliberately retain
their scan/file provenance so the experiment runner can audit the split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from types import MappingProxyType
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd


CHANNELS: tuple[int, ...] = (5180, 5200, 5220, 5240, 5260, 5280, 5300, 5320)
STATE_FEATURES: tuple[str, ...] = (
    "snr_mean",
    "snr_std",
    "rssi_mean",
    "rssi_std",
    "noise_mean",
)
DEFAULT_SCAN_SPLITS: Mapping[str, tuple[int, ...]] = MappingProxyType(
    {
        "train": tuple(range(0, 7)),
        "val": (7,),
        "test": (8, 9),
    }
)

_REQUIRED_COLUMNS = frozenset({"freq1", "snr", "rssi", "noise"})
_FILENAME_RE = re.compile(
    r"^samples_chamber_(?P<jammer>\d+)MHz_"
    r"(?P<distance>\d+)cm_(?P<power>-?\d+)dBm_"
    r"(?P<scan>\d+)\.csv$",
    flags=re.IGNORECASE,
)


class DatasetFormatError(ValueError):
    """Raised when the raw CSV matrix is incomplete or malformed."""


class DataLeakageError(ValueError):
    """Raised when a source scan/file is assigned to multiple splits."""


@dataclass(frozen=True, order=True)
class TrajectoryKey:
    """Identity of one measured trajectory."""

    scan_id: int
    jammer_channel: int


@dataclass(frozen=True)
class ScanSplit:
    """Windowed trajectories and their immutable provenance for one split.

    Each trajectory has shape ``[n_windows, 40]``.  The 40 columns are ordered
    by :attr:`channels`; each channel contributes the five entries in
    :data:`STATE_FEATURES`.
    """

    name: str
    channels: tuple[int, ...]
    trajectories: Mapping[TrajectoryKey, np.ndarray]
    window_starts: Mapping[TrajectoryKey, np.ndarray]
    source_files: Mapping[TrajectoryKey, Path]
    window_size: int
    stride: int
    distance_cm: int
    power_dbm: int

    @property
    def scan_ids(self) -> tuple[int, ...]:
        return tuple(sorted({key.scan_id for key in self.trajectories}))

    @property
    def jammer_channels(self) -> tuple[int, ...]:
        return tuple(sorted({key.jammer_channel for key in self.trajectories}))

    @property
    def observation_size(self) -> int:
        return len(self.channels) * len(STATE_FEATURES)

    @property
    def observations(self) -> np.ndarray:
        """All trajectories concatenated in deterministic key order."""

        arrays = [self.trajectories[key] for key in sorted(self.trajectories)]
        if not arrays:
            return np.empty((0, self.observation_size), dtype=np.float32)
        return np.concatenate(arrays, axis=0)

    @property
    def metadata(self) -> pd.DataFrame:
        """One provenance row per observation (useful for exported audits)."""

        rows: list[pd.DataFrame] = []
        for key in sorted(self.trajectories):
            starts = self.window_starts[key]
            rows.append(
                pd.DataFrame(
                    {
                        "split": self.name,
                        "scan_id": key.scan_id,
                        "jammer_channel": key.jammer_channel,
                        "window_start": starts,
                        "source_file": str(self.source_files[key]),
                        "distance_cm": self.distance_cm,
                        "power_dbm": self.power_dbm,
                    }
                )
            )
        if not rows:
            return pd.DataFrame(
                columns=(
                    "split",
                    "scan_id",
                    "jammer_channel",
                    "window_start",
                    "source_file",
                    "distance_cm",
                    "power_dbm",
                )
            )
        return pd.concat(rows, ignore_index=True)

    def trajectory(self, scan_id: int, jammer_channel: int) -> np.ndarray:
        """Return a trajectory without copying it."""

        return self.trajectories[TrajectoryKey(int(scan_id), int(jammer_channel))]

    def common_length(self, scan_id: int) -> int:
        """Number of time positions available for every jammer channel."""

        lengths = [
            len(self.trajectory(scan_id, jammer_channel))
            for jammer_channel in self.channels
        ]
        return min(lengths)

    def iter_trajectories(
        self,
    ) -> Iterator[tuple[TrajectoryKey, np.ndarray, np.ndarray, Path]]:
        for key in sorted(self.trajectories):
            yield (
                key,
                self.trajectories[key],
                self.window_starts[key],
                self.source_files[key],
            )


@dataclass(frozen=True)
class DatasetBundle:
    """All leakage-safe splits for one fixed physical condition."""

    splits: Mapping[str, ScanSplit]
    channels: tuple[int, ...]
    window_size: int
    stride: int
    distance_cm: int
    power_dbm: int
    raw_dir: Path

    def __getitem__(self, split: str) -> ScanSplit:
        return self.splits[split]

    def __iter__(self) -> Iterator[str]:
        return iter(self.splits)

    def __len__(self) -> int:
        return len(self.splits)

    def keys(self):
        return self.splits.keys()

    def values(self):
        return self.splits.values()

    def items(self):
        return self.splits.items()

    @property
    def train(self) -> ScanSplit:
        return self.splits["train"]

    @property
    def val(self) -> ScanSplit:
        return self.splits["val"]

    @property
    def test(self) -> ScanSplit:
        return self.splits["test"]


def _normalise_splits(
    split_scan_ids: Mapping[str, Sequence[int]],
) -> dict[str, tuple[int, ...]]:
    if not split_scan_ids:
        raise ValueError("split_scan_ids must not be empty")

    normalised: dict[str, tuple[int, ...]] = {}
    owner: dict[int, str] = {}
    for split_name, raw_ids in split_scan_ids.items():
        ids = tuple(int(scan_id) for scan_id in raw_ids)
        if len(ids) != len(set(ids)):
            raise DataLeakageError(f"duplicate scan id inside split {split_name!r}")
        for scan_id in ids:
            if scan_id in owner:
                raise DataLeakageError(
                    f"scan {scan_id} occurs in both {owner[scan_id]!r} "
                    f"and {split_name!r}"
                )
            owner[scan_id] = str(split_name)
        normalised[str(split_name)] = tuple(sorted(ids))
    return normalised


def discover_scan_files(
    raw_dir: str | Path,
    *,
    distance_cm: int = 20,
    power_dbm: int = 10,
    channels: Sequence[int] | None = CHANNELS,
) -> dict[TrajectoryKey, Path]:
    """Discover files for exactly one physical condition.

    Files from other distances/powers are intentionally ignored.  This makes
    it possible to train on 20 cm/10 dBm and later construct 40 cm/10 dBm or
    20 cm/5 dBm bundles for frozen-policy OOD evaluation.
    """

    directory = Path(raw_dir).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"raw data directory does not exist: {directory}")

    allowed = None if channels is None else {int(channel) for channel in channels}
    found: dict[TrajectoryKey, Path] = {}
    for path in sorted(directory.glob("*.csv")):
        match = _FILENAME_RE.match(path.name)
        if match is None:
            continue
        jammer = int(match.group("jammer"))
        if int(match.group("distance")) != int(distance_cm):
            continue
        if int(match.group("power")) != int(power_dbm):
            continue
        if allowed is not None and jammer not in allowed:
            continue
        key = TrajectoryKey(int(match.group("scan")), jammer)
        if key in found:
            raise DatasetFormatError(
                f"duplicate files for scan={key.scan_id}, jammer={jammer}: "
                f"{found[key]} and {path}"
            )
        found[key] = path.resolve()

    if not found:
        raise FileNotFoundError(
            f"no matching CSVs in {directory} for "
            f"distance_cm={distance_cm}, power_dbm={power_dbm}"
        )
    return found


def _finite_window_mean_std(
    values: np.ndarray,
    *,
    window_size: int,
    starts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """NaN-aware rolling sample statistics without warning on empty windows."""

    windows = np.lib.stride_tricks.sliding_window_view(values, window_size)[starts]
    finite = np.isfinite(windows)
    counts = finite.sum(axis=1)
    safe_values = np.where(finite, windows, 0.0)
    sums = safe_values.sum(axis=1, dtype=np.float64)
    means = np.divide(
        sums,
        counts,
        out=np.zeros_like(sums, dtype=np.float64),
        where=counts > 0,
    )
    centred = np.where(finite, windows - means[:, None], 0.0)
    sum_squares = np.square(centred, dtype=np.float64).sum(axis=1)
    stds = np.sqrt(
        np.divide(
            sum_squares,
            counts - 1,
            out=np.zeros_like(sum_squares),
            where=counts > 1,
        )
    )
    return means, stds


def featurize_scan_file(
    csv_path: str | Path,
    *,
    channels: Sequence[int] = CHANNELS,
    window_size: int = 32,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert one CSV into independent, within-file feature windows.

    Returns ``(features, window_starts)``.  ``features`` has shape
    ``[n_windows, len(channels) * 5]``.  Invalid/non-finite SNR values and SNR
    values outside [-100, 100] are omitted from SNR statistics; non-finite RSSI
    and noise entries are omitted from their corresponding statistics.  A
    statistic with no finite samples is represented as 0 rather than allowing
    NaN/Inf to enter a neural network.
    """

    if int(window_size) <= 0:
        raise ValueError("window_size must be positive")
    if int(stride) <= 0:
        raise ValueError("stride must be positive")
    channel_tuple = tuple(int(channel) for channel in channels)
    if not channel_tuple:
        raise ValueError("channels must not be empty")

    path = Path(csv_path).expanduser().resolve()
    frame = pd.read_csv(path)
    missing_columns = sorted(_REQUIRED_COLUMNS.difference(frame.columns))
    if missing_columns:
        raise DatasetFormatError(
            f"{path.name} is missing required columns: {missing_columns}"
        )

    numeric = frame.loc[:, ["freq1", "snr", "rssi", "noise"]].copy()
    for column in numeric.columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")

    by_channel: dict[int, np.ndarray] = {}
    for channel in channel_tuple:
        selected = numeric.loc[
            numeric["freq1"] == channel, ["snr", "rssi", "noise"]
        ].to_numpy(dtype=np.float64, copy=True)
        if len(selected) < window_size:
            raise DatasetFormatError(
                f"{path.name}: channel {channel} has {len(selected)} rows, "
                f"fewer than window_size={window_size}"
            )
        selected[~np.isfinite(selected)] = np.nan
        invalid_snr = (selected[:, 0] < -100.0) | (selected[:, 0] > 100.0)
        selected[invalid_snr, 0] = np.nan
        by_channel[channel] = selected

    # A state must represent the same within-file time position on all channels.
    # Truncating to the shortest channel is explicit and never crosses a file.
    common_rows = min(len(values) for values in by_channel.values())
    starts = np.arange(
        0,
        common_rows - int(window_size) + 1,
        int(stride),
        dtype=np.int64,
    )
    if not len(starts):
        raise DatasetFormatError(f"{path.name}: no complete windows")

    per_channel: list[np.ndarray] = []
    for channel in channel_tuple:
        values = by_channel[channel][:common_rows]
        snr_mean, snr_std = _finite_window_mean_std(
            values[:, 0], window_size=int(window_size), starts=starts
        )
        rssi_mean, rssi_std = _finite_window_mean_std(
            values[:, 1], window_size=int(window_size), starts=starts
        )
        noise_mean, _ = _finite_window_mean_std(
            values[:, 2], window_size=int(window_size), starts=starts
        )
        per_channel.append(
            np.column_stack(
                (snr_mean, snr_std, rssi_mean, rssi_std, noise_mean)
            )
        )

    features = np.stack(per_channel, axis=1).reshape(len(starts), -1)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features.astype(np.float32, copy=False), starts


def build_dataset(
    raw_dir: str | Path,
    *,
    split_scan_ids: Mapping[str, Sequence[int]] = DEFAULT_SCAN_SPLITS,
    channels: Sequence[int] = CHANNELS,
    window_size: int = 32,
    stride: int = 1,
    distance_cm: int = 20,
    power_dbm: int = 10,
) -> DatasetBundle:
    """Load one condition and apply a scan-level split before windowing."""

    normalised_splits = _normalise_splits(split_scan_ids)
    channel_tuple = tuple(int(channel) for channel in channels)
    files = discover_scan_files(
        raw_dir,
        distance_cm=distance_cm,
        power_dbm=power_dbm,
        channels=channel_tuple,
    )

    requested_scan_ids = {
        scan_id for scan_ids in normalised_splits.values() for scan_id in scan_ids
    }
    missing = [
        TrajectoryKey(scan_id, jammer_channel)
        for scan_id in sorted(requested_scan_ids)
        for jammer_channel in channel_tuple
        if TrajectoryKey(scan_id, jammer_channel) not in files
    ]
    if missing:
        preview = ", ".join(
            f"(scan={key.scan_id}, jammer={key.jammer_channel})"
            for key in missing[:8]
        )
        suffix = " ..." if len(missing) > 8 else ""
        raise DatasetFormatError(
            f"incomplete raw-file matrix for {distance_cm}cm/{power_dbm}dBm: "
            f"{preview}{suffix}"
        )

    # Feature extraction is keyed by the already assigned source scan.  There
    # is intentionally no pre-split concatenation or global sliding window.
    built_splits: dict[str, ScanSplit] = {}
    for split_name, scan_ids in normalised_splits.items():
        trajectories: dict[TrajectoryKey, np.ndarray] = {}
        starts_by_key: dict[TrajectoryKey, np.ndarray] = {}
        sources: dict[TrajectoryKey, Path] = {}
        for scan_id in scan_ids:
            for jammer_channel in channel_tuple:
                key = TrajectoryKey(scan_id, jammer_channel)
                features, starts = featurize_scan_file(
                    files[key],
                    channels=channel_tuple,
                    window_size=window_size,
                    stride=stride,
                )
                features.setflags(write=False)
                starts.setflags(write=False)
                trajectories[key] = features
                starts_by_key[key] = starts
                sources[key] = files[key]
        built_splits[split_name] = ScanSplit(
            name=split_name,
            channels=channel_tuple,
            trajectories=MappingProxyType(trajectories),
            window_starts=MappingProxyType(starts_by_key),
            source_files=MappingProxyType(sources),
            window_size=int(window_size),
            stride=int(stride),
            distance_cm=int(distance_cm),
            power_dbm=int(power_dbm),
        )

    bundle = DatasetBundle(
        splits=MappingProxyType(built_splits),
        channels=channel_tuple,
        window_size=int(window_size),
        stride=int(stride),
        distance_cm=int(distance_cm),
        power_dbm=int(power_dbm),
        raw_dir=Path(raw_dir).expanduser().resolve(),
    )
    assert_no_scan_leakage(bundle)
    return bundle


def build_ood_dataset(
    raw_dir: str | Path,
    *,
    distance_cm: int,
    power_dbm: int,
    scan_ids: Sequence[int] = tuple(range(10)),
    channels: Sequence[int] = CHANNELS,
    window_size: int = 32,
    stride: int = 1,
) -> DatasetBundle:
    """Build a single evaluation-only split for a held-out condition.

    The split is deliberately named ``"ood"``.  The threshold baseline refuses
    to fit on that name, providing a mechanical guard against adapting a policy
    or heuristic after inspecting 40 cm/10 dBm or 20 cm/5 dBm results.
    """

    return build_dataset(
        raw_dir,
        split_scan_ids={"ood": tuple(int(scan_id) for scan_id in scan_ids)},
        channels=channels,
        window_size=window_size,
        stride=stride,
        distance_cm=distance_cm,
        power_dbm=power_dbm,
    )


def assert_no_scan_leakage(
    dataset: DatasetBundle | Mapping[str, ScanSplit],
) -> None:
    """Raise :class:`DataLeakageError` if any scan/file crosses a split."""

    splits = dataset.splits if isinstance(dataset, DatasetBundle) else dataset
    scan_owner: dict[int, str] = {}
    source_owner: dict[Path, str] = {}
    for split_name, split in splits.items():
        for key, trajectory, starts, source in split.iter_trajectories():
            if key.scan_id in scan_owner and scan_owner[key.scan_id] != split_name:
                raise DataLeakageError(
                    f"scan {key.scan_id} appears in {scan_owner[key.scan_id]!r} "
                    f"and {split_name!r}"
                )
            scan_owner[key.scan_id] = split_name

            resolved = source.resolve()
            if resolved in source_owner and source_owner[resolved] != split_name:
                raise DataLeakageError(
                    f"source file {resolved} appears in "
                    f"{source_owner[resolved]!r} and {split_name!r}"
                )
            source_owner[resolved] = split_name

            if trajectory.ndim != 2 or trajectory.shape[1] != split.observation_size:
                raise DatasetFormatError(
                    f"invalid trajectory shape {trajectory.shape} for {key}"
                )
            if len(trajectory) != len(starts):
                raise DatasetFormatError(
                    f"feature/provenance length mismatch for {key}"
                )
            if len(starts) and (
                np.any(np.diff(starts) != split.stride) or starts[0] != 0
            ):
                raise DatasetFormatError(
                    f"non-contiguous window provenance for {key}"
                )


# Clear aliases for runners/notebooks that use a loader-oriented name.
load_dataset = build_dataset
load_spectral_dataset = build_dataset
build_evaluation_dataset = build_ood_dataset


__all__ = [
    "CHANNELS",
    "STATE_FEATURES",
    "DEFAULT_SCAN_SPLITS",
    "DatasetFormatError",
    "DataLeakageError",
    "TrajectoryKey",
    "ScanSplit",
    "DatasetBundle",
    "discover_scan_files",
    "featurize_scan_file",
    "build_dataset",
    "build_ood_dataset",
    "build_evaluation_dataset",
    "load_dataset",
    "load_spectral_dataset",
    "assert_no_scan_leakage",
]
