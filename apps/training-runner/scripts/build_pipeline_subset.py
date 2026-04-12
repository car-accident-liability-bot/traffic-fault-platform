from __future__ import annotations

import sys
from pathlib import Path


# 주석:
# 설치 전에도 기존 scripts 진입점을 바로 실행할 수 있도록
# src 패키지 경로를 보수적으로 sys.path 에 추가한다.
REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINING_SRC = REPO_ROOT / "apps" / "training-runner" / "src"
AI_CORE_SRC = REPO_ROOT / "packages" / "ai-core" / "src"

for candidate in (TRAINING_SRC, AI_CORE_SRC):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from training_runner.cli.build_pipeline_subset import main


if __name__ == "__main__":
    main()
