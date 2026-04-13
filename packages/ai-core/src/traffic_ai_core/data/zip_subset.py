from __future__ import annotations

import csv
import hashlib
import html.parser
import io
import json
import math
import random
import re
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import quote, unquote, urljoin, urlparse, urlunparse

DEFAULT_DATASET_BASE_URL = "https://data.taeo-dev.com/dataset/traffic"
ZIP_SUFFIX = ".zip"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
LABEL_EXTENSIONS = {".json"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
DEFAULT_HTTP_TIMEOUT_SECONDS = 30
DEFAULT_HTTP_CHUNK_SIZE = 8 * 1024 * 1024
HTTP_USER_AGENT = "traffic-fault-platform/0.1 (+subset-builder)"
DEFAULT_REMOTE_SCAN_WORKERS = 2
DEFAULT_LABEL_INSPECTION_WORKERS = 2
DEFAULT_CACHE_DIR = Path("data/cache")
ZIP_INVENTORY_CACHE_PATH = DEFAULT_CACHE_DIR / "zip_inventory_cache.json"
LABEL_INSPECTION_CACHE_PATH = DEFAULT_CACHE_DIR / "label_inspection_cache.json"
CACHE_FORMAT_VERSION = 1
_REMOTE_FILE_SIZE_CACHE: dict[str, int] = {}
_REMOTE_RANGE_SUPPORT_CACHE: dict[str, bool] = {}

# 주석:
# - 실제 공개 데이터셋 JSON 스키마가 문서로 고정돼 있지 않아서,
#   전처리 필드는 한국어/영문 후보 키를 넉넉히 받아서 재귀 탐색한다.
# - 후보 키를 못 찾는 경우 전체를 오판해 버리지 않도록 값 존재 여부를 보수적으로 판정한다.
FAULT_RATIO_KEY_CANDIDATES = (
    "과실비율",
    "과실율",
    "fault_ratio",
    "faultRatio",
    "fault_rate",
    "faultRate",
    "liability_ratio",
    "liabilityRatio",
    "negligence_ratio",
    "negligenceRatio",
)
ROAD_TYPE_KEY_CANDIDATES = (
    "도로유형",
    "도로 형태",
    "road_type",
    "roadType",
    "road_category",
    "roadCategory",
)
CASE_CODE_KEY_CANDIDATES = (
    "사고유형코드",
    "사고 유형 코드",
    "과실유형코드",
    "과실 유형 코드",
    "case_code",
    "caseCode",
    "accident_type_code",
    "accidentTypeCode",
    "liability_type_code",
    "liabilityTypeCode",
)
CORE_LABEL_KEY_CANDIDATES = (
    *FAULT_RATIO_KEY_CANDIDATES,
    *ROAD_TYPE_KEY_CANDIDATES,
    *CASE_CODE_KEY_CANDIDATES,
)
AMBIGUOUS_TEXT_VALUES = {
    "",
    "-",
    "--",
    "n/a",
    "na",
    "nan",
    "none",
    "null",
    "unknown",
    "undefined",
    "미상",
    "불명",
    "알수없음",
    "알 수 없음",
    "기타",
    "기타/미상",
    "unknown/other",
    "미정",
    "없음",
    "기재없음",
    "기재 없음",
}


@dataclass(frozen=True)
class DatasetMetadata:
    """manifest와 summary에 공통으로 들어가는 공용 기준 메타데이터."""

    dataset_base_url: str
    split: str
    raw_dir_name: str
    label_dir_name: str


@dataclass(frozen=True)
class ZipInventory:
    """ZIP 내부 확장자 기준 인벤토리 정보."""

    zip_name: str
    zip_local_path: Path | None
    zip_access_url: str | None
    sample_members: dict[str, str]
    sample_ids: frozenset[str]
    relevant_member_count: int
    image_member_count: int
    other_member_count: int


@dataclass(frozen=True)
class CategoryZipBundle:
    """매칭된 raw ZIP / label ZIP의 런타임 묶음 정보."""

    category: str
    raw_zip_name: str
    label_zip_name: str
    raw_zip_local_path: Path | None
    label_zip_local_path: Path | None
    raw_zip_access_url: str | None
    label_zip_access_url: str | None
    raw_zip_relative_path: str
    label_zip_relative_path: str
    raw_zip_public_url: str
    label_zip_public_url: str
    raw_sample_members: dict[str, str]
    label_sample_members: dict[str, str]


@dataclass(frozen=True)
class SelectedSample:
    """샘플링과 추출에 사용하는 런타임 레코드.

    주석:
    - local path는 로컬 디렉터리 모드에서만 필요하다.
    - URL 모드에서는 access_url을 사용해 원격 ZIP에서 직접 읽는다.
    - manifest 저장은 아래 ManifestRecord로 별도 변환한다.
    """

    split: str
    category: str
    sample_id: str
    raw_zip_name: str
    label_zip_name: str
    raw_zip_local_path: Path | None
    label_zip_local_path: Path | None
    raw_zip_access_url: str | None
    label_zip_access_url: str | None
    raw_zip_relative_path: str
    label_zip_relative_path: str
    raw_zip_public_url: str
    label_zip_public_url: str
    video_member_name: str
    label_member_name: str


@dataclass(frozen=True)
class ManifestRecord:
    """팀 공용 manifest 저장용 레코드.

    주석:
    - 절대경로는 의도적으로 제외했다.
    - 상대경로 / URL / 파일명 중심으로만 저장한다.
    """

    split: str
    category: str
    sample_id: str
    raw_zip_relative_path: str
    label_zip_relative_path: str
    raw_zip_public_url: str
    label_zip_public_url: str
    raw_zip_name: str
    label_zip_name: str
    video_member_name: str
    label_member_name: str


@dataclass(frozen=True)
class LabelQualityInspection:
    """라벨 JSON 품질 점검 결과."""

    has_valid_fault_ratio: bool
    road_type: str | None
    case_code: str | None
    has_minimum_required_fields: bool


@dataclass(frozen=True)
class LabelQualityOptions:
    """subset 생성 전 라벨 품질 필터 옵션."""

    enabled: bool = True
    require_fault_ratio: bool = True
    require_road_type: bool = True
    require_case_code: bool = True
    rare_case_code_min_frequency: int = 2


def _stable_cache_key(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _load_cache_entries(cache_path: Path) -> dict[str, Any]:
    try:
        with cache_path.open("r", encoding="utf-8") as cache_file:
            payload = json.load(cache_file)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}

    if not isinstance(payload, dict):
        return {}
    if payload.get("version") != CACHE_FORMAT_VERSION:
        return {}

    entries = payload.get("entries")
    if not isinstance(entries, dict):
        return {}
    return entries


def _save_cache_entries(cache_path: Path, entries: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    payload = {
        "version": CACHE_FORMAT_VERSION,
        "entries": entries,
    }
    with temp_path.open("w", encoding="utf-8") as cache_file:
        json.dump(payload, cache_file, ensure_ascii=False, indent=2, sort_keys=True)
    temp_path.replace(cache_path)


def _get_source_signature(local_path: Path | None, access_url: str | None) -> dict[str, Any]:
    if local_path is not None:
        stat_result = local_path.stat()
        return {
            "kind": "local",
            "path": str(local_path.resolve()),
            "size": stat_result.st_size,
            "mtime_ns": stat_result.st_mtime_ns,
        }

    if access_url is not None:
        return {
            "kind": "remote",
            "url": access_url,
            "size": _fetch_remote_file_size(access_url, timeout_seconds=DEFAULT_HTTP_TIMEOUT_SECONDS),
        }

    raise ValueError("캐시 키 생성을 위한 source 정보가 없습니다.")


def _build_zip_inventory_cache_key(
    *,
    zip_name: str,
    local_path: Path | None,
    access_url: str | None,
    allowed_extensions: set[str],
) -> str:
    payload = {
        "type": "zip_inventory",
        "zip_name": zip_name,
        "allowed_extensions": sorted(allowed_extensions),
        "source": _get_source_signature(local_path, access_url),
    }
    return _stable_cache_key(payload)


def _serialize_zip_inventory_cache_entry(inventory: ZipInventory) -> dict[str, Any]:
    return {
        "zip_name": inventory.zip_name,
        "sample_members": inventory.sample_members,
        "relevant_member_count": inventory.relevant_member_count,
        "image_member_count": inventory.image_member_count,
        "other_member_count": inventory.other_member_count,
    }


def _deserialize_zip_inventory_cache_entry(
    entry: Any,
    *,
    zip_name: str,
    local_path: Path | None,
    access_url: str | None,
) -> ZipInventory | None:
    if not isinstance(entry, dict):
        return None

    sample_members = entry.get("sample_members")
    if not isinstance(sample_members, dict):
        return None

    normalized_members = {str(key): str(value) for key, value in sample_members.items()}
    return ZipInventory(
        zip_name=zip_name,
        zip_local_path=local_path,
        zip_access_url=access_url,
        sample_members=normalized_members,
        sample_ids=frozenset(normalized_members.keys()),
        relevant_member_count=int(entry.get("relevant_member_count", len(normalized_members))),
        image_member_count=int(entry.get("image_member_count", 0)),
        other_member_count=int(entry.get("other_member_count", 0)),
    )


def _build_label_inspection_cache_key(
    *,
    local_path: Path | None,
    access_url: str | None,
    zip_name: str,
    member_name: str,
) -> str:
    payload = {
        "type": "label_inspection",
        "zip_name": zip_name,
        "member_name": member_name,
        "source": _get_source_signature(local_path, access_url),
    }
    return _stable_cache_key(payload)


def _serialize_label_inspection_cache_entry(inspection: LabelQualityInspection) -> dict[str, Any]:
    return {
        "has_valid_fault_ratio": inspection.has_valid_fault_ratio,
        "road_type": inspection.road_type,
        "case_code": inspection.case_code,
        "has_minimum_required_fields": inspection.has_minimum_required_fields,
    }


def _deserialize_label_inspection_cache_entry(entry: Any) -> LabelQualityInspection | None:
    if not isinstance(entry, dict):
        return None

    required_keys = {
        "has_valid_fault_ratio",
        "road_type",
        "case_code",
        "has_minimum_required_fields",
    }
    if not required_keys.issubset(entry.keys()):
        return None

    return LabelQualityInspection(
        has_valid_fault_ratio=bool(entry["has_valid_fault_ratio"]),
        road_type=entry["road_type"] if entry["road_type"] is None else str(entry["road_type"]),
        case_code=entry["case_code"] if entry["case_code"] is None else str(entry["case_code"]),
        has_minimum_required_fields=bool(entry["has_minimum_required_fields"]),
    )


def _accumulate_label_inspection_counters(
    inspection: LabelQualityInspection,
    road_type_counter: Counter[str],
    case_code_counter: Counter[str],
) -> None:
    if inspection.road_type is not None:
        road_type_counter[inspection.road_type] += 1
    if inspection.case_code is not None:
        case_code_counter[inspection.case_code] += 1


class _DirectoryIndexParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        attrs_dict = dict(attrs)
        href = attrs_dict.get("href")
        if href:
            self.hrefs.append(href)


class RemoteHttpRangeReader(io.RawIOBase):
    """HTTP Range 요청으로 ZIP을 seekable 파일처럼 읽는 최소 구현."""

    def __init__(
        self,
        url: str,
        *,
        chunk_size: int = DEFAULT_HTTP_CHUNK_SIZE,
        timeout_seconds: int = DEFAULT_HTTP_TIMEOUT_SECONDS,
    ) -> None:
        super().__init__()
        self._url = url
        self._chunk_size = max(64 * 1024, chunk_size)
        self._timeout_seconds = timeout_seconds
        self._size = _fetch_remote_file_size(url, timeout_seconds=timeout_seconds)
        self._position = 0
        self._chunk_cache: dict[int, bytes] = {}

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            next_position = offset
        elif whence == io.SEEK_CUR:
            next_position = self._position + offset
        elif whence == io.SEEK_END:
            next_position = self._size + offset
        else:
            raise ValueError(f"지원하지 않는 whence 값입니다: {whence}")

        if next_position < 0:
            raise ValueError("음수 위치로 seek 할 수 없습니다.")

        self._position = min(next_position, self._size)
        return self._position

    def read(self, size: int = -1) -> bytes:
        if self.closed:
            raise ValueError("닫힌 파일입니다.")

        if size is None or size < 0:
            size = self._size - self._position

        if size <= 0 or self._position >= self._size:
            return b""

        start = self._position
        end_exclusive = min(self._size, self._position + size)
        data = self._read_range(start, end_exclusive)
        self._position = end_exclusive
        return data

    def readinto(self, buffer: bytearray | memoryview) -> int:
        data = self.read(len(buffer))
        data_length = len(data)
        buffer[:data_length] = data
        return data_length

    def close(self) -> None:
        self._chunk_cache.clear()
        super().close()

    def _read_range(self, start: int, end_exclusive: int) -> bytes:
        parts: list[bytes] = []
        current = start

        while current < end_exclusive:
            chunk_index = current // self._chunk_size
            chunk_start = chunk_index * self._chunk_size
            chunk = self._get_chunk(chunk_index)
            offset = current - chunk_start
            take = min(len(chunk) - offset, end_exclusive - current)
            if take <= 0:
                raise OSError(
                    f"원격 ZIP 청크를 읽지 못했습니다. url={self._url}, position={current}"
                )
            parts.append(chunk[offset : offset + take])
            current += take

        return b"".join(parts)

    def _get_chunk(self, chunk_index: int) -> bytes:
        cached = self._chunk_cache.get(chunk_index)
        if cached is not None:
            return cached

        start = chunk_index * self._chunk_size
        end = min(self._size, start + self._chunk_size) - 1
        chunk = _fetch_remote_range_bytes(
            self._url,
            start,
            end,
            timeout_seconds=self._timeout_seconds,
        )
        expected_size = end - start + 1

        if len(chunk) < expected_size:
            raise OSError(
                "HTTP Range 응답 길이가 예상보다 짧습니다. "
                f"url={self._url}, requested={expected_size}, actual={len(chunk)}"
            )
        if len(chunk) > expected_size:
            chunk = chunk[:expected_size]

        self._chunk_cache[chunk_index] = chunk
        return chunk

def _build_public_url(dataset_base_url: str, relative_path: PurePosixPath) -> str:
    encoded_parts = [quote(part) for part in relative_path.parts]
    return f"{dataset_base_url.rstrip('/')}/{'/'.join(encoded_parts)}"


def _normalize_directory_url(directory_url: str) -> str:
    stripped = directory_url.strip()
    if not stripped:
        raise ValueError("비어 있는 URL은 사용할 수 없습니다.")
    if not stripped.endswith("/"):
        stripped += "/"
    return stripped


def _encode_url_path(url: str) -> str:
    parsed = urlparse(url)
    encoded_path = quote(unquote(parsed.path), safe="/")
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            encoded_path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )


def _build_request(url: str, *, method: str = "GET", headers: dict[str, str] | None = None) -> urllib.request.Request:
    request_headers = {"User-Agent": HTTP_USER_AGENT}
    if headers:
        request_headers.update(headers)
    return urllib.request.Request(_encode_url_path(url), headers=request_headers, method=method)


def _sanitize_cache_filename(name: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z._-가-힣]+", "_", name).strip("._")
    return sanitized or "remote.zip"


def _remote_url_supports_range(url: str, *, timeout_seconds: int = DEFAULT_HTTP_TIMEOUT_SECONDS) -> bool:
    cache_key = _encode_url_path(url)
    cached = _REMOTE_RANGE_SUPPORT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    request = _build_request(url, headers={"Range": "bytes=0-0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status_code = getattr(response, "status", None)
            content_range = response.headers.get("Content-Range")
            response.read(1)
            supported = bool(content_range or status_code == 206)
    except urllib.error.HTTPError as exc:
        if exc.code in {400, 405, 416, 501}:
            supported = False
        else:
            supported = False
    except (urllib.error.URLError, OSError, ValueError):
        supported = False

    _REMOTE_RANGE_SUPPORT_CACHE[cache_key] = supported
    return supported


def _download_remote_file_to_temporary_path(
    url: str,
    *,
    timeout_seconds: int = DEFAULT_HTTP_TIMEOUT_SECONDS,
) -> Path:
    """Range 미지원 환경에서만 사용하는 임시 로컬 ZIP 파일을 만든다.

    주석:
    - 영구 캐시 디렉터리를 사용하지 않는다.
    - subset 실행 중에만 임시 파일을 만들고, 사용이 끝나면 즉시 삭제한다.
    - 그래서 대용량 remote_zip_file_cache 디렉터리가 계속 쌓이지 않는다.
    """
    expected_size = _fetch_remote_file_size(url, timeout_seconds=timeout_seconds)
    encoded_url = _encode_url_path(url)
    parsed = urlparse(encoded_url)
    original_name = unquote(PurePosixPath(parsed.path).name) or "remote.zip"
    suffix = Path(original_name).suffix or ".zip"

    with tempfile.NamedTemporaryFile(prefix="traffic_remote_zip_", suffix=suffix, delete=False) as temp_file:
        temp_path = Path(temp_file.name)
        request = _build_request(url)
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            shutil.copyfileobj(response, temp_file, length=1024 * 1024)

    actual_size = temp_path.stat().st_size
    if expected_size > 0 and actual_size != expected_size:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise OSError(
            "원격 ZIP 전체 다운로드 길이가 예상과 다릅니다. "
            f"url={url}, expected={expected_size}, actual={actual_size}"
        )

    return temp_path


def _determine_remote_worker_count(urls: Iterable[str | None], default_workers: int, item_count: int) -> int:
    max_candidates = max(1, item_count)
    remote_urls = [url for url in urls if url]
    if not remote_urls:
        return min(default_workers, max_candidates)

    sample_url = remote_urls[0]
    if not _remote_url_supports_range(sample_url, timeout_seconds=DEFAULT_HTTP_TIMEOUT_SECONDS):
        return 1

    return min(default_workers, max_candidates)


def _infer_remote_directory_identity(directory_url: str) -> tuple[str, str]:
    normalized_url = _normalize_directory_url(directory_url)
    path_parts = [part for part in urlparse(normalized_url).path.split("/") if part]
    if len(path_parts) < 2:
        raise ValueError(
            "URL 경로에서 split 과 디렉터리 이름을 추론할 수 없습니다. "
            f"url={normalized_url}"
        )

    split_name = unquote(path_parts[-2])
    dir_name = unquote(path_parts[-1])
    return split_name, dir_name


def derive_dataset_metadata(
    raw_dir: Path | None = None,
    label_dir: Path | None = None,
    *,
    raw_url: str | None = None,
    label_url: str | None = None,
    dataset_base_url: str = DEFAULT_DATASET_BASE_URL,
    split_name: str | None = None,
    raw_dir_name: str | None = None,
    label_dir_name: str | None = None,
) -> DatasetMetadata:
    """로컬 디렉터리 또는 공용 URL로부터 manifest용 메타데이터를 만든다."""
    using_local_dirs = raw_dir is not None or label_dir is not None
    using_urls = raw_url is not None or label_url is not None

    if using_local_dirs and using_urls:
        raise ValueError("로컬 디렉터리와 URL 입력을 동시에 사용할 수 없습니다.")

    if raw_dir is not None and label_dir is not None:
        resolved_raw_dir = raw_dir.expanduser().resolve()
        resolved_label_dir = label_dir.expanduser().resolve()

        if not resolved_raw_dir.is_dir():
            raise FileNotFoundError(f"raw 디렉터리를 찾을 수 없습니다: {resolved_raw_dir}")
        if not resolved_label_dir.is_dir():
            raise FileNotFoundError(f"label 디렉터리를 찾을 수 없습니다: {resolved_label_dir}")

        inferred_split = resolved_raw_dir.parent.name
        inferred_raw_dir_name = resolved_raw_dir.name
        inferred_label_dir_name = resolved_label_dir.name

        if resolved_raw_dir.parent != resolved_label_dir.parent and split_name is None:
            raise ValueError(
                "raw-dir 과 label-dir 의 상위 split 디렉터리가 다릅니다. "
                "다른 구조를 사용한다면 --split-name 옵션으로 명시해 주세요."
            )
    elif raw_url is not None and label_url is not None:
        inferred_split, inferred_raw_dir_name = _infer_remote_directory_identity(raw_url)
        label_split, inferred_label_dir_name = _infer_remote_directory_identity(label_url)
        if inferred_split != label_split and split_name is None:
            raise ValueError(
                "raw-url 과 label-url 의 split 경로가 다릅니다. "
                "다른 구조를 사용한다면 --split-name 옵션으로 명시해 주세요."
            )
    else:
        raise ValueError("raw/label 입력은 로컬 디렉터리 한 쌍 또는 URL 한 쌍으로 모두 제공해야 합니다.")

    effective_split = split_name or inferred_split
    effective_raw_dir_name = raw_dir_name or inferred_raw_dir_name
    effective_label_dir_name = label_dir_name or inferred_label_dir_name

    return DatasetMetadata(
        dataset_base_url=dataset_base_url.rstrip("/"),
        split=effective_split,
        raw_dir_name=effective_raw_dir_name,
        label_dir_name=effective_label_dir_name,
    )


def _decode_http_body(body: bytes, content_type: str | None) -> str:
    charset = None
    if content_type:
        match = re.search(r"charset=([\w\-]+)", content_type, re.IGNORECASE)
        if match:
            charset = match.group(1)
    for candidate in [charset, "utf-8", "cp949", "euc-kr"]:
        if not candidate:
            continue
        try:
            return body.decode(candidate)
        except UnicodeDecodeError:
            continue
    return body.decode("utf-8", errors="replace")


def _load_directory_index_html(directory_url: str, *, timeout_seconds: int = DEFAULT_HTTP_TIMEOUT_SECONDS) -> str:
    normalized_url = _normalize_directory_url(directory_url)
    request = _build_request(normalized_url)
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        body = response.read()
        content_type = response.headers.get("Content-Type")
    return _decode_http_body(body, content_type)


def _list_remote_zip_files(
    directory_url: str,
    *,
    timeout_seconds: int = DEFAULT_HTTP_TIMEOUT_SECONDS,
) -> list[tuple[str, str]]:
    parser = _DirectoryIndexParser()
    parser.feed(_load_directory_index_html(directory_url, timeout_seconds=timeout_seconds))

    files: dict[str, str] = {}
    normalized_base = _normalize_directory_url(directory_url)

    for href in parser.hrefs:
        if href in {"../", "./", "#"}:
            continue

        absolute_url = _encode_url_path(urljoin(normalized_base, href))
        parsed = urlparse(absolute_url)
        file_name = unquote(PurePosixPath(parsed.path).name)
        if not file_name or not file_name.lower().endswith(ZIP_SUFFIX):
            continue
        files[file_name] = absolute_url

    return sorted(files.items(), key=lambda item: item[0])


def _fetch_remote_file_size(url: str, *, timeout_seconds: int) -> int:
    cached_size = _REMOTE_FILE_SIZE_CACHE.get(url)
    if cached_size is not None:
        return cached_size

    try:
        request = _build_request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                file_size = int(content_length)
                _REMOTE_FILE_SIZE_CACHE[url] = file_size
                return file_size
    except (urllib.error.URLError, ValueError, OSError):
        pass

    try:
        request = _build_request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            content_range = response.headers.get("Content-Range")
            if content_range:
                match = re.match(r"bytes\s+\d+-\d+/(\d+)", content_range)
                if match:
                    file_size = int(match.group(1))
                    _REMOTE_FILE_SIZE_CACHE[url] = file_size
                    return file_size

            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                file_size = int(content_length)
                _REMOTE_FILE_SIZE_CACHE[url] = file_size
                return file_size
    except urllib.error.HTTPError:
        pass

    request = _build_request(url)
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            file_size = int(content_length)
            _REMOTE_FILE_SIZE_CACHE[url] = file_size
            return file_size

    raise OSError(f"원격 파일 크기를 확인할 수 없습니다: {url}")


def _fetch_remote_range_bytes(
    url: str,
    start: int,
    end: int,
    *,
    timeout_seconds: int,
) -> bytes:
    if start < 0 or end < start:
        raise ValueError(f"유효하지 않은 Range 요청입니다: start={start}, end={end}")

    request = _build_request(url, headers={"Range": f"bytes={start}-{end}"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status_code = getattr(response, "status", None)
            content_range = response.headers.get("Content-Range")
            if content_range or status_code == 206:
                return response.read()
            if status_code == 200:
                raise OSError(
                    "원격 서버가 HTTP Range 요청을 무시했습니다. "
                    f"url={url}, range={start}-{end}"
                )
            return response.read()
    except urllib.error.HTTPError as exc:
        if exc.code in {400, 405, 416, 501}:
            raise OSError(
                "원격 서버가 HTTP Range 요청을 처리하지 않습니다. "
                f"url={url}, status={exc.code}"
            ) from exc
        raise


def _create_zip_file(source: Path | io.BufferedIOBase | RemoteHttpRangeReader) -> zipfile.ZipFile:
    return zipfile.ZipFile(source, mode="r", metadata_encoding="cp949")


@contextmanager
def _open_zip_file(
    *,
    local_path: Path | None,
    access_url: str | None,
) -> Iterable[zipfile.ZipFile]:
    if local_path is not None:
        with _create_zip_file(local_path) as zip_file:
            yield zip_file
        return

    if access_url is None:
        raise ValueError("ZIP 접근 경로가 없습니다. local_path 또는 access_url 중 하나는 필요합니다.")

    if _remote_url_supports_range(access_url, timeout_seconds=DEFAULT_HTTP_TIMEOUT_SECONDS):
        remote_reader = RemoteHttpRangeReader(access_url)
        try:
            with _create_zip_file(remote_reader) as zip_file:
                yield zip_file
        finally:
            remote_reader.close()
        return

    downloaded_path = _download_remote_file_to_temporary_path(
        access_url,
        timeout_seconds=DEFAULT_HTTP_TIMEOUT_SECONDS,
    )
    try:
        with _create_zip_file(downloaded_path) as zip_file:
            yield zip_file
    finally:
        try:
            downloaded_path.unlink(missing_ok=True)
        except OSError:
            pass


def _member_selection_score(member_name: str) -> tuple[int, int, int, str]:
    normalized = member_name.replace("\\", "/")
    path = PurePosixPath(normalized)
    parts = path.parts

    is_macosx = any(part == "__MACOSX" for part in parts)
    is_hidden = any(part.startswith(".") for part in parts)
    depth = len(parts)
    return (
        0 if is_macosx else 1,
        0 if is_hidden else 1,
        -depth,
        normalized,
    )


def _select_preferred_member(existing_member: str, candidate_member: str) -> str:
    if _member_selection_score(candidate_member) > _member_selection_score(existing_member):
        return candidate_member
    return existing_member


def _scan_single_zip_inventory(
    *,
    zip_name: str,
    local_path: Path | None,
    access_url: str | None,
    allowed_extensions: set[str],
) -> ZipInventory:
    sample_members: dict[str, str] = {}
    image_member_count = 0
    other_member_count = 0

    with _open_zip_file(local_path=local_path, access_url=access_url) as zip_file:
        for member_name in zip_file.namelist():
            if member_name.endswith("/"):
                continue

            normalized_member_name = member_name.replace("\\", "/")
            member_path = PurePosixPath(normalized_member_name)
            extension = member_path.suffix.lower()

            if extension in allowed_extensions:
                sample_id = member_path.stem
                existing_member = sample_members.get(sample_id)
                if existing_member is None:
                    sample_members[sample_id] = normalized_member_name
                else:
                    sample_members[sample_id] = _select_preferred_member(
                        existing_member,
                        normalized_member_name,
                    )
                continue

            if extension in IMAGE_EXTENSIONS:
                image_member_count += 1
            else:
                other_member_count += 1

    return ZipInventory(
        zip_name=zip_name,
        zip_local_path=local_path,
        zip_access_url=access_url,
        sample_members=sample_members,
        sample_ids=frozenset(sample_members.keys()),
        relevant_member_count=len(sample_members),
        image_member_count=image_member_count,
        other_member_count=other_member_count,
    )


def _scan_local_zip_inventory(directory: Path, allowed_extensions: set[str]) -> list[ZipInventory]:
    inventories: list[ZipInventory] = []

    for zip_path in sorted(directory.glob(f"*{ZIP_SUFFIX}")):
        inventory = _scan_single_zip_inventory(
            zip_name=zip_path.name,
            local_path=zip_path,
            access_url=None,
            allowed_extensions=allowed_extensions,
        )
        if not inventory.sample_members:
            continue
        inventories.append(inventory)

    return inventories


def _scan_remote_zip_inventory(directory_url: str, allowed_extensions: set[str]) -> list[ZipInventory]:
    zip_files = _list_remote_zip_files(directory_url)
    if not zip_files:
        return []

    cache_entries = _load_cache_entries(ZIP_INVENTORY_CACHE_PATH)
    cache_updates: dict[str, Any] = {}
    inventories: list[ZipInventory] = []

    def worker(zip_name: str, zip_url: str) -> tuple[ZipInventory | None, str | None, dict[str, Any] | None]:
        cache_key = _build_zip_inventory_cache_key(
            zip_name=zip_name,
            local_path=None,
            access_url=zip_url,
            allowed_extensions=allowed_extensions,
        )
        cached_inventory = _deserialize_zip_inventory_cache_entry(
            cache_entries.get(cache_key),
            zip_name=zip_name,
            local_path=None,
            access_url=zip_url,
        )
        if cached_inventory is not None:
            return cached_inventory, None, None

        inventory = _scan_single_zip_inventory(
            zip_name=zip_name,
            local_path=None,
            access_url=zip_url,
            allowed_extensions=allowed_extensions,
        )
        return inventory, cache_key, _serialize_zip_inventory_cache_entry(inventory)

    max_workers = _determine_remote_worker_count([zip_url for _zip_name, zip_url in zip_files], DEFAULT_REMOTE_SCAN_WORKERS, len(zip_files))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(worker, zip_name, zip_url): (zip_name, zip_url)
            for zip_name, zip_url in zip_files
        }

        for future in as_completed(future_map):
            inventory, cache_key, cache_entry = future.result()
            if inventory is None or not inventory.sample_members:
                continue
            inventories.append(inventory)
            if cache_key is not None and cache_entry is not None:
                cache_updates[cache_key] = cache_entry

    if cache_updates:
        cache_entries.update(cache_updates)
        _save_cache_entries(ZIP_INVENTORY_CACHE_PATH, cache_entries)

    return sorted(inventories, key=lambda item: item.zip_name)


def _tokenize_stem(stem: str) -> list[str]:
    return [token for token in re.split(r"[\s_\-]+", stem) if token]


def _common_suffix_tokens(left: list[str], right: list[str]) -> list[str]:
    result: list[str] = []
    for left_token, right_token in zip(reversed(left), reversed(right)):
        if left_token != right_token:
            break
        result.append(left_token)
    return list(reversed(result))


def _common_prefix_tokens(left: list[str], right: list[str]) -> list[str]:
    result: list[str] = []
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        result.append(left_token)
    return result


def _strip_generic_category_tokens(tokens: list[str]) -> list[str]:
    generic_tokens = {
        "raw",
        "label",
        "labels",
        "video",
        "videos",
        "json",
        "annotation",
        "annotations",
        "annot",
        "ts",
        "tl",
        "차대차",
        "영상",
        "이미지",
    }
    return [token for token in tokens if token.lower() not in generic_tokens]


def _derive_category(raw_zip_name: str, label_zip_name: str) -> str:
    raw_stem = Path(raw_zip_name).stem
    label_stem = Path(label_zip_name).stem

    if raw_stem == label_stem:
        return raw_stem

    raw_tokens = _tokenize_stem(raw_stem)
    label_tokens = _tokenize_stem(label_stem)

    common_suffix = _strip_generic_category_tokens(_common_suffix_tokens(raw_tokens, label_tokens))
    if common_suffix:
        return "_".join(common_suffix)

    common_prefix = _strip_generic_category_tokens(_common_prefix_tokens(raw_tokens, label_tokens))
    if common_prefix:
        return "_".join(common_prefix)

    fallback_tokens = _strip_generic_category_tokens(raw_tokens)
    if fallback_tokens:
        return "_".join(fallback_tokens)

    return raw_stem


def _maximum_weight_assignment(weights: list[list[int]]) -> list[int]:
    """헝가리안 알고리즘으로 최대 가중치 1:1 매칭을 구한다."""
    size = len(weights)
    if size == 0:
        return []

    max_weight = max(max(row) for row in weights)
    costs = [[max_weight - value for value in row] for row in weights]

    u = [0] * (size + 1)
    v = [0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)

    for left_index in range(1, size + 1):
        p[0] = left_index
        minv = [math.inf] * (size + 1)
        used = [False] * (size + 1)
        column_index = 0

        while True:
            used[column_index] = True
            row_index = p[column_index]
            delta = math.inf
            next_column = 0

            for candidate_column in range(1, size + 1):
                if used[candidate_column]:
                    continue

                current = costs[row_index - 1][candidate_column - 1] - u[row_index] - v[candidate_column]
                if current < minv[candidate_column]:
                    minv[candidate_column] = current
                    way[candidate_column] = column_index

                if minv[candidate_column] < delta:
                    delta = minv[candidate_column]
                    next_column = candidate_column

            for candidate_column in range(size + 1):
                if used[candidate_column]:
                    u[p[candidate_column]] += delta
                    v[candidate_column] -= delta
                else:
                    minv[candidate_column] -= delta

            column_index = next_column
            if p[column_index] == 0:
                break

        while True:
            previous_column = way[column_index]
            p[column_index] = p[previous_column]
            column_index = previous_column
            if column_index == 0:
                break

    assignment = [-1] * size
    for column_index in range(1, size + 1):
        if p[column_index] != 0:
            assignment[p[column_index] - 1] = column_index - 1

    return assignment



def _build_fast_zip_pair_key(zip_name: str) -> str:
    stem = Path(zip_name).stem
    if stem.startswith("TS_") or stem.startswith("TL_"):
        return stem[3:]
    return stem


def _pair_zip_inventories_by_overlap(
    raw_inventories: list[ZipInventory],
    label_inventories: list[ZipInventory],
) -> list[tuple[ZipInventory, ZipInventory, int]]:
    if len(raw_inventories) != len(label_inventories):
        raise ValueError(
            "relevant ZIP 개수가 raw/label 사이에 다릅니다. "
            f"raw={len(raw_inventories)}, label={len(label_inventories)}"
        )

    if not raw_inventories:
        return []

    weights: list[list[int]] = []
    for raw_inventory in raw_inventories:
        raw_ids = raw_inventory.sample_ids
        row: list[int] = []
        for label_inventory in label_inventories:
            row.append(len(raw_ids & label_inventory.sample_ids))
        weights.append(row)

    unmatched_raw = [
        raw_inventory.zip_name
        for raw_inventory, row in zip(raw_inventories, weights)
        if max(row, default=0) <= 0
    ]
    unmatched_label = [
        label_inventory.zip_name
        for column_index, label_inventory in enumerate(label_inventories)
        if max(weights[row_index][column_index] for row_index in range(len(weights))) <= 0
    ]

    if unmatched_raw:
        raise ValueError(
            "basename overlap으로 매칭되지 않은 raw ZIP이 있습니다: "
            f"{sorted(unmatched_raw)}"
        )
    if unmatched_label:
        raise ValueError(
            "basename overlap으로 매칭되지 않은 label ZIP이 있습니다: "
            f"{sorted(unmatched_label)}"
        )

    assignment = _maximum_weight_assignment(weights)

    matched_pairs: list[tuple[ZipInventory, ZipInventory, int]] = []
    for raw_index, label_index in enumerate(assignment):
        if label_index < 0:
            raise ValueError("ZIP 1:1 매칭 계산에 실패했습니다.")

        overlap_count = weights[raw_index][label_index]
        if overlap_count <= 0:
            raise ValueError(
                "ZIP 1:1 매칭 결과에 overlap 0인 쌍이 포함됐습니다. "
                "ZIP 내부 basename 구성을 다시 확인해 주세요."
            )

        matched_pairs.append(
            (
                raw_inventories[raw_index],
                label_inventories[label_index],
                overlap_count,
            )
        )

    return matched_pairs


def _pair_zip_inventories(
    raw_inventories: list[ZipInventory],
    label_inventories: list[ZipInventory],
) -> list[tuple[ZipInventory, ZipInventory, int]]:
    if len(raw_inventories) != len(label_inventories):
        raise ValueError(
            "relevant ZIP 개수가 raw/label 사이에 다릅니다. "
            f"raw={len(raw_inventories)}, label={len(label_inventories)}"
        )

    if not raw_inventories:
        return []

    raw_groups: dict[str, list[ZipInventory]] = {}
    label_groups: dict[str, list[ZipInventory]] = {}

    for inventory in raw_inventories:
        raw_groups.setdefault(_build_fast_zip_pair_key(inventory.zip_name), []).append(inventory)
    for inventory in label_inventories:
        label_groups.setdefault(_build_fast_zip_pair_key(inventory.zip_name), []).append(inventory)

    matched_pairs: list[tuple[ZipInventory, ZipInventory, int]] = []
    matched_raw_names: set[str] = set()
    matched_label_names: set[str] = set()

    for pair_key in sorted(set(raw_groups) & set(label_groups)):
        raw_group = raw_groups[pair_key]
        label_group = label_groups[pair_key]

        if len(raw_group) != 1 or len(label_group) != 1:
            continue

        raw_inventory = raw_group[0]
        label_inventory = label_group[0]
        overlap_count = len(raw_inventory.sample_ids & label_inventory.sample_ids)
        if overlap_count <= 0:
            continue

        matched_pairs.append((raw_inventory, label_inventory, overlap_count))
        matched_raw_names.add(raw_inventory.zip_name)
        matched_label_names.add(label_inventory.zip_name)

    remaining_raw = [inventory for inventory in raw_inventories if inventory.zip_name not in matched_raw_names]
    remaining_label = [inventory for inventory in label_inventories if inventory.zip_name not in matched_label_names]

    if remaining_raw or remaining_label:
        matched_pairs.extend(_pair_zip_inventories_by_overlap(remaining_raw, remaining_label))

    return matched_pairs


def build_category_bundles(

    metadata: DatasetMetadata,
    *,
    raw_dir: Path | None = None,
    label_dir: Path | None = None,
    raw_url: str | None = None,
    label_url: str | None = None,
) -> list[CategoryZipBundle]:
    using_local_dirs = raw_dir is not None and label_dir is not None
    using_urls = raw_url is not None and label_url is not None

    if using_local_dirs == using_urls:
        raise ValueError("로컬 디렉터리 또는 URL 모드 중 하나만 선택해야 합니다.")

    if using_local_dirs:
        raw_inventories = _scan_local_zip_inventory(raw_dir, VIDEO_EXTENSIONS)
        label_inventories = _scan_local_zip_inventory(label_dir, LABEL_EXTENSIONS)
        raw_source_display = str(raw_dir)
        label_source_display = str(label_dir)
    else:
        normalized_raw_url = _normalize_directory_url(raw_url)
        normalized_label_url = _normalize_directory_url(label_url)
        raw_inventories = _scan_remote_zip_inventory(normalized_raw_url, VIDEO_EXTENSIONS)
        label_inventories = _scan_remote_zip_inventory(normalized_label_url, LABEL_EXTENSIONS)
        raw_source_display = normalized_raw_url
        label_source_display = normalized_label_url

    if not raw_inventories:
        raise ValueError(f"사용 가능한 영상 ZIP을 찾지 못했습니다: {raw_source_display}")
    if not label_inventories:
        raise ValueError(f"사용 가능한 JSON ZIP을 찾지 못했습니다: {label_source_display}")

    matched_pairs = _pair_zip_inventories(raw_inventories, label_inventories)

    bundles: list[CategoryZipBundle] = []
    used_categories: set[str] = set()

    for raw_inventory, label_inventory, _ in matched_pairs:
        category = _derive_category(raw_inventory.zip_name, label_inventory.zip_name)
        if category in used_categories:
            category = f"{category}__{Path(raw_inventory.zip_name).stem}"
        used_categories.add(category)

        raw_zip_relative_path = PurePosixPath(
            metadata.split,
            metadata.raw_dir_name,
            raw_inventory.zip_name,
        )
        label_zip_relative_path = PurePosixPath(
            metadata.split,
            metadata.label_dir_name,
            label_inventory.zip_name,
        )

        bundles.append(
            CategoryZipBundle(
                category=category,
                raw_zip_name=raw_inventory.zip_name,
                label_zip_name=label_inventory.zip_name,
                raw_zip_local_path=raw_inventory.zip_local_path,
                label_zip_local_path=label_inventory.zip_local_path,
                raw_zip_access_url=raw_inventory.zip_access_url,
                label_zip_access_url=label_inventory.zip_access_url,
                raw_zip_relative_path=str(raw_zip_relative_path),
                label_zip_relative_path=str(label_zip_relative_path),
                raw_zip_public_url=(
                    raw_inventory.zip_access_url
                    if raw_inventory.zip_access_url is not None
                    else _build_public_url(metadata.dataset_base_url, raw_zip_relative_path)
                ),
                label_zip_public_url=(
                    label_inventory.zip_access_url
                    if label_inventory.zip_access_url is not None
                    else _build_public_url(metadata.dataset_base_url, label_zip_relative_path)
                ),
                raw_sample_members=raw_inventory.sample_members,
                label_sample_members=label_inventory.sample_members,
            )
        )

    return sorted(bundles, key=lambda item: item.category)


def build_records_for_bundle(
    bundle: CategoryZipBundle,
    split: str,
) -> tuple[list[SelectedSample], dict[str, int | str]]:
    raw_members = bundle.raw_sample_members
    label_members = bundle.label_sample_members

    raw_ids = set(raw_members.keys())
    label_ids = set(label_members.keys())

    matched_ids = sorted(raw_ids & label_ids)
    raw_only_ids = sorted(raw_ids - label_ids)
    label_only_ids = sorted(label_ids - raw_ids)

    records = [
        SelectedSample(
            split=split,
            category=bundle.category,
            sample_id=sample_id,
            raw_zip_name=bundle.raw_zip_name,
            label_zip_name=bundle.label_zip_name,
            raw_zip_local_path=bundle.raw_zip_local_path,
            label_zip_local_path=bundle.label_zip_local_path,
            raw_zip_access_url=bundle.raw_zip_access_url,
            label_zip_access_url=bundle.label_zip_access_url,
            raw_zip_relative_path=bundle.raw_zip_relative_path,
            label_zip_relative_path=bundle.label_zip_relative_path,
            raw_zip_public_url=bundle.raw_zip_public_url,
            label_zip_public_url=bundle.label_zip_public_url,
            video_member_name=raw_members[sample_id],
            label_member_name=label_members[sample_id],
        )
        for sample_id in matched_ids
    ]

    diagnostics: dict[str, int | str] = {
        "category": bundle.category,
        "raw_zip_name": bundle.raw_zip_name,
        "label_zip_name": bundle.label_zip_name,
        "raw_count": len(raw_ids),
        "label_count": len(label_ids),
        "matched_count": len(matched_ids),
        "raw_only_count": len(raw_only_ids),
        "label_only_count": len(label_only_ids),
    }
    return records, diagnostics


def build_all_category_records(
    metadata: DatasetMetadata,
    *,
    raw_dir: Path | None = None,
    label_dir: Path | None = None,
    raw_url: str | None = None,
    label_url: str | None = None,
) -> tuple[dict[str, list[SelectedSample]], list[dict[str, int | str]]]:
    bundles = build_category_bundles(
        metadata=metadata,
        raw_dir=raw_dir,
        label_dir=label_dir,
        raw_url=raw_url,
        label_url=label_url,
    )

    category_records: dict[str, list[SelectedSample]] = {}
    diagnostics: list[dict[str, int | str]] = []

    for bundle in bundles:
        records, stats = build_records_for_bundle(bundle=bundle, split=metadata.split)
        category_records[bundle.category] = records
        diagnostics.append(stats)

    return category_records, diagnostics


def _normalize_key(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z가-힣]+", "", text).lower()


def _normalize_text_value(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, bool):
        return str(value).lower()

    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return str(value)

    text = str(value).strip()
    if not text:
        return None
    return text


def _is_ambiguous_text(text: str | None) -> bool:
    if text is None:
        return True
    normalized = re.sub(r"\s+", "", text).lower()
    return normalized in {_normalize_key(item) for item in AMBIGUOUS_TEXT_VALUES} or normalized == ""



def _normalize_scalar_iter(node: Any) -> Iterable[str]:
    if isinstance(node, dict):
        for value in node.values():
            yield from _normalize_scalar_iter(value)
        return
    if isinstance(node, list):
        for item in node:
            yield from _normalize_scalar_iter(item)
        return

    text = _normalize_text_value(node)
    if text is not None:
        yield text


def _analyze_label_payload(payload: Any) -> LabelQualityInspection:
    normalized_fault_keys = {_normalize_key(candidate) for candidate in FAULT_RATIO_KEY_CANDIDATES}
    normalized_road_keys = {_normalize_key(candidate) for candidate in ROAD_TYPE_KEY_CANDIDATES}
    normalized_case_keys = {_normalize_key(candidate) for candidate in CASE_CODE_KEY_CANDIDATES}
    normalized_core_keys = {_normalize_key(candidate) for candidate in CORE_LABEL_KEY_CANDIDATES}
    normalized_ambiguous_values = {_normalize_key(item) for item in AMBIGUOUS_TEXT_VALUES}

    has_valid_fault_ratio = False
    road_type: str | None = None
    case_code: str | None = None
    has_minimum_required_fields = False

    def walk(node: Any) -> None:
        nonlocal has_valid_fault_ratio, road_type, case_code, has_minimum_required_fields

        if isinstance(node, dict):
            for key, value in node.items():
                normalized_key = _normalize_key(str(key))

                if normalized_key in normalized_core_keys:
                    has_minimum_required_fields = True

                if normalized_key in normalized_fault_keys and not has_valid_fault_ratio:
                    for text in _normalize_scalar_iter(value):
                        normalized_text = re.sub(r"\s+", "", text).lower()
                        if normalized_text in normalized_ambiguous_values:
                            continue
                        if normalized_text in {"nan", "nannan", "none", "null"}:
                            continue
                        if re.search(r"\d", text):
                            has_valid_fault_ratio = True
                            break

                if normalized_key in normalized_road_keys and road_type is None:
                    for text in _normalize_scalar_iter(value):
                        if _is_ambiguous_text(text):
                            continue
                        road_type = text
                        break

                if normalized_key in normalized_case_keys and case_code is None:
                    for text in _normalize_scalar_iter(value):
                        if _is_ambiguous_text(text):
                            continue
                        case_code = text
                        break

                walk(value)
            return

        if isinstance(node, list):
            for item in node:
                walk(item)

    if isinstance(payload, (dict, list)) and payload:
        walk(payload)

    return LabelQualityInspection(
        has_valid_fault_ratio=has_valid_fault_ratio,
        road_type=road_type,
        case_code=case_code,
        has_minimum_required_fields=has_minimum_required_fields,
    )


def _inspect_label_payload(payload: Any) -> LabelQualityInspection:
    return _analyze_label_payload(payload)


def _read_label_payload_from_zip(zip_file: zipfile.ZipFile, member_name: str) -> Any:
    with zip_file.open(member_name) as source:
        return json.load(source)


def _group_records_by_label_zip(
    category_records: dict[str, list[SelectedSample]],
) -> dict[tuple[Path | None, str | None, str], list[tuple[str, SelectedSample]]]:
    grouped: dict[tuple[Path | None, str | None, str], list[tuple[str, SelectedSample]]] = {}
    for category, records in category_records.items():
        for record in records:
            key = (record.label_zip_local_path, record.label_zip_access_url, record.label_zip_name)
            grouped.setdefault(key, []).append((category, record))
    return grouped


def _inspect_label_group_with_cache(
    *,
    local_path: Path | None,
    access_url: str | None,
    zip_name: str,
    group_items: list[tuple[str, SelectedSample]],
    cache_entries: dict[str, Any],
) -> tuple[
    dict[tuple[str, str], LabelQualityInspection | None],
    dict[tuple[str, str], str],
    Counter[str],
    Counter[str],
    int,
    dict[str, Any],
]:
    inspections: dict[tuple[str, str], LabelQualityInspection | None] = {}
    failure_reasons: dict[tuple[str, str], str] = {}
    road_type_counter: Counter[str] = Counter()
    case_code_counter: Counter[str] = Counter()
    cache_updates: dict[str, Any] = {}
    total_before_filter = len(group_items)

    pending_items: list[tuple[str, SelectedSample, str]] = []

    for category, record in group_items:
        sample_key = (category, record.sample_id)
        cache_key = _build_label_inspection_cache_key(
            local_path=local_path,
            access_url=access_url,
            zip_name=zip_name,
            member_name=record.label_member_name,
        )
        cached_inspection = _deserialize_label_inspection_cache_entry(cache_entries.get(cache_key))
        if cached_inspection is not None:
            inspections[sample_key] = cached_inspection
            _accumulate_label_inspection_counters(cached_inspection, road_type_counter, case_code_counter)
            continue
        pending_items.append((category, record, cache_key))

    if not pending_items:
        return (
            inspections,
            failure_reasons,
            road_type_counter,
            case_code_counter,
            total_before_filter,
            cache_updates,
        )

    try:
        with _open_zip_file(local_path=local_path, access_url=access_url) as zip_file:
            for category, record, cache_key in pending_items:
                sample_key = (category, record.sample_id)
                try:
                    payload = _read_label_payload_from_zip(zip_file, record.label_member_name)
                    inspection = _inspect_label_payload(payload)
                    inspections[sample_key] = inspection
                    _accumulate_label_inspection_counters(inspection, road_type_counter, case_code_counter)
                    cache_updates[cache_key] = _serialize_label_inspection_cache_entry(inspection)
                except (OSError, ValueError, json.JSONDecodeError, zipfile.BadZipFile, KeyError) as exc:
                    inspections[sample_key] = None
                    failure_reasons[sample_key] = f"label_json_read_error:{type(exc).__name__}"
    except (OSError, ValueError, zipfile.BadZipFile, KeyError) as exc:
        for category, record, _cache_key in pending_items:
            sample_key = (category, record.sample_id)
            inspections[sample_key] = None
            failure_reasons[sample_key] = f"label_zip_open_error:{type(exc).__name__}"

    return (
        inspections,
        failure_reasons,
        road_type_counter,
        case_code_counter,
        total_before_filter,
        cache_updates,
    )


def preprocess_category_records(
    category_records: dict[str, list[SelectedSample]],
    options: LabelQualityOptions,
) -> tuple[dict[str, list[SelectedSample]], dict[str, Any]]:
    if not options.enabled:
        total_records = sum(len(records) for records in category_records.values())
        return category_records, {
            "enabled": False,
            "total_before_filter": total_records,
            "total_after_filter": total_records,
            "excluded_total": 0,
            "excluded_by_reason": {},
            "road_type_frequency": {},
            "case_code_frequency": {},
        }

    inspections: dict[tuple[str, str], LabelQualityInspection | None] = {}
    failure_reasons: dict[tuple[str, str], str] = {}
    road_type_counter: Counter[str] = Counter()
    case_code_counter: Counter[str] = Counter()
    total_before_filter = 0

    grouped_records = _group_records_by_label_zip(category_records)
    cache_entries = _load_cache_entries(LABEL_INSPECTION_CACHE_PATH)
    cache_updates: dict[str, Any] = {}

    group_entries = list(grouped_records.items())
    max_workers = _determine_remote_worker_count([access_url for (local_path, access_url, _zip_name), _group_items in group_entries], DEFAULT_LABEL_INSPECTION_WORKERS, len(group_entries))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                _inspect_label_group_with_cache,
                local_path=local_path,
                access_url=access_url,
                zip_name=zip_name,
                group_items=group_items,
                cache_entries=cache_entries,
            ): (local_path, access_url, zip_name)
            for (local_path, access_url, zip_name), group_items in group_entries
        }

        for future in as_completed(future_map):
            (
                group_inspections,
                group_failure_reasons,
                group_road_type_counter,
                group_case_code_counter,
                group_total_before_filter,
                group_cache_updates,
            ) = future.result()

            total_before_filter += group_total_before_filter
            inspections.update(group_inspections)
            failure_reasons.update(group_failure_reasons)
            road_type_counter.update(group_road_type_counter)
            case_code_counter.update(group_case_code_counter)
            cache_updates.update(group_cache_updates)

    if cache_updates:
        cache_entries.update(cache_updates)
        _save_cache_entries(LABEL_INSPECTION_CACHE_PATH, cache_entries)

    filtered_category_records: dict[str, list[SelectedSample]] = {}
    excluded_by_reason: Counter[str] = Counter()

    for category, records in category_records.items():
        kept_records: list[SelectedSample] = []

        for record in records:
            key = (category, record.sample_id)

            if key in failure_reasons:
                excluded_by_reason[failure_reasons[key]] += 1
                continue

            inspection = inspections.get(key)
            if inspection is None:
                excluded_by_reason["label_json_read_error"] += 1
                continue

            exclusion_reason: str | None = None

            if not inspection.has_minimum_required_fields:
                exclusion_reason = "missing_required_label_fields"
            elif options.require_fault_ratio and not inspection.has_valid_fault_ratio:
                exclusion_reason = "missing_or_invalid_fault_ratio"
            elif options.require_road_type and inspection.road_type is None:
                exclusion_reason = "missing_or_ambiguous_road_type"
            elif options.require_case_code and inspection.case_code is None:
                exclusion_reason = "missing_or_invalid_case_code"
            elif (
                inspection.case_code is not None
                and options.rare_case_code_min_frequency > 1
                and case_code_counter[inspection.case_code] < options.rare_case_code_min_frequency
            ):
                exclusion_reason = "rare_case_code"

            if exclusion_reason is not None:
                excluded_by_reason[exclusion_reason] += 1
                continue

            kept_records.append(record)

        filtered_category_records[category] = kept_records

    total_after_filter = sum(len(records) for records in filtered_category_records.values())

    summary: dict[str, Any] = {
        "enabled": True,
        "options": {
            "require_fault_ratio": options.require_fault_ratio,
            "require_road_type": options.require_road_type,
            "require_case_code": options.require_case_code,
            "rare_case_code_min_frequency": options.rare_case_code_min_frequency,
        },
        "total_before_filter": total_before_filter,
        "total_after_filter": total_after_filter,
        "excluded_total": total_before_filter - total_after_filter,
        "excluded_by_reason": dict(sorted(excluded_by_reason.items())),
        "road_type_frequency": dict(sorted(road_type_counter.items())),
        "case_code_frequency": dict(sorted(case_code_counter.items())),
    }

    return filtered_category_records, summary


def sample_per_category(
    category_records: dict[str, list[SelectedSample]],
    per_category: int,
    seed: int,
) -> list[SelectedSample]:
    if per_category <= 0:
        raise ValueError("per_category는 1 이상이어야 합니다.")

    rng = random.Random(seed)
    selected_records: list[SelectedSample] = []

    for category in sorted(category_records.keys()):
        records = sorted(category_records[category], key=lambda item: item.sample_id)

        if len(records) <= per_category:
            chosen = records
        else:
            chosen = sorted(rng.sample(records, per_category), key=lambda item: item.sample_id)

        selected_records.extend(chosen)

    return selected_records


def to_manifest_records(records: Iterable[SelectedSample]) -> list[ManifestRecord]:
    return [
        ManifestRecord(
            split=record.split,
            category=record.category,
            sample_id=record.sample_id,
            raw_zip_relative_path=record.raw_zip_relative_path,
            label_zip_relative_path=record.label_zip_relative_path,
            raw_zip_public_url=record.raw_zip_public_url,
            label_zip_public_url=record.label_zip_public_url,
            raw_zip_name=record.raw_zip_name,
            label_zip_name=record.label_zip_name,
            video_member_name=record.video_member_name,
            label_member_name=record.label_member_name,
        )
        for record in records
    ]


def write_manifest_csv(records: Iterable[SelectedSample], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_records = to_manifest_records(records)

    fieldnames = [
        "split",
        "category",
        "sample_id",
        "raw_zip_relative_path",
        "label_zip_relative_path",
        "raw_zip_public_url",
        "label_zip_public_url",
        "raw_zip_name",
        "label_zip_name",
        "video_member_name",
        "label_member_name",
    ]

    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in manifest_records:
            writer.writerow(asdict(record))


def write_manifest_json(records: Iterable[SelectedSample], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_records = to_manifest_records(records)

    payload = {
        "records": [asdict(record) for record in manifest_records],
        "total_records": len(manifest_records),
    }

    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, ensure_ascii=False, indent=2)


def write_summary_json(
    output_path: Path,
    selected_records: list[SelectedSample],
    diagnostics: list[dict[str, int | str]],
    metadata: DatasetMetadata,
    per_category: int,
    seed: int,
    preprocessing_summary: dict[str, Any] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    selected_count_by_category = Counter(record.category for record in selected_records)

    summary = {
        "dataset_base_url": metadata.dataset_base_url,
        "split": metadata.split,
        "raw_dir_name": metadata.raw_dir_name,
        "label_dir_name": metadata.label_dir_name,
        "per_category": per_category,
        "seed": seed,
        "total_selected": len(selected_records),
        "selected_count_by_category": dict(sorted(selected_count_by_category.items())),
        "diagnostics": diagnostics,
    }
    if preprocessing_summary is not None:
        summary["preprocessing"] = preprocessing_summary

    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, ensure_ascii=False, indent=2)


def _group_records_by_zip_source(
    records: list[SelectedSample],
    *,
    kind: str,
) -> dict[tuple[Path | None, str | None, str], list[SelectedSample]]:
    grouped: dict[tuple[Path | None, str | None, str], list[SelectedSample]] = {}

    for record in records:
        if kind == "raw":
            key = (record.raw_zip_local_path, record.raw_zip_access_url, record.raw_zip_name)
        elif kind == "label":
            key = (record.label_zip_local_path, record.label_zip_access_url, record.label_zip_name)
        else:
            raise ValueError(f"지원하지 않는 ZIP 그룹 kind입니다: {kind}")
        grouped.setdefault(key, []).append(record)

    return grouped


def extract_selected_files(records: list[SelectedSample], output_root: Path) -> None:
    raw_root = output_root / "raw"
    label_root = output_root / "label"
    raw_root.mkdir(parents=True, exist_ok=True)
    label_root.mkdir(parents=True, exist_ok=True)

    raw_group = _group_records_by_zip_source(records, kind="raw")
    label_group = _group_records_by_zip_source(records, kind="label")

    for (local_path, access_url, _zip_name), group_records in raw_group.items():
        with _open_zip_file(local_path=local_path, access_url=access_url) as zip_file:
            for record in group_records:
                category_dir = raw_root / record.category
                category_dir.mkdir(parents=True, exist_ok=True)
                target_path = category_dir / Path(record.video_member_name).name

                with zip_file.open(record.video_member_name) as source, target_path.open("wb") as target:
                    shutil.copyfileobj(source, target)

    for (local_path, access_url, _zip_name), group_records in label_group.items():
        with _open_zip_file(local_path=local_path, access_url=access_url) as zip_file:
            for record in group_records:
                category_dir = label_root / record.category
                category_dir.mkdir(parents=True, exist_ok=True)
                target_path = category_dir / Path(record.label_member_name).name

                with zip_file.open(record.label_member_name) as source, target_path.open("wb") as target:
                    shutil.copyfileobj(source, target)
