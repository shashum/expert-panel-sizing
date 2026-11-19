#!/usr/bin/env bash
# 전체 재현 경로: 원자료(data/raw/survey.xlsx)가 있을 때 모든 분석, 모의실험, 그림을 순서대로 실행한다.
# 원자료 없이 공개 파생 자료만으로 확인하려면: python code/30_reproduce_from_public_data.py
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs/logs outputs/figures

if [ ! -f "${SURVEY_XLSX:-data/raw/survey.xlsx}" ]; then
  echo "원자료가 없습니다. data/README.md를 보세요. 공개 자료 경로만 실행합니다."
  python code/30_reproduce_from_public_data.py | tee outputs/logs/30_reproduce_from_public_data.txt
  exit 0
fi

for s in 00_derive_public_data 01_alpha 02_resampling 03_icc 04_reproducibility 05_variance_groups \
         06_lme 07_sensitivity 08_split_distributions 09_supplementary_checks \
         10_sim_reproduce 11_simulations 12_sim_additions 20_figures 30_reproduce_from_public_data; do
  echo "== $s"
  python "code/$s.py" > "outputs/logs/$s.txt" 2>&1
done

if command -v node >/dev/null && node -e "require('pptxgenjs')" 2>/dev/null; then
  node code/21_figure5.js > outputs/logs/21_figure5.txt 2>&1
else
  echo "그림 5는 Node.js와 pptxgenjs가 필요합니다: npm install pptxgenjs && node code/21_figure5.js"
fi
echo "완료: outputs/ 와 outputs/logs/ 를 확인하세요."
