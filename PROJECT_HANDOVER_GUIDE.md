# 🎯 LLM 벤치마크 시각화 도구 - 프로젝트 인수인계 가이드

## 📋 목차
1. [프로젝트 개요](#1-프로젝트-개요)
2. [시스템 아키텍처](#2-시스템-아키텍처)
3. [데이터 구조](#3-데이터-구조)
4. [설치 및 실행](#4-설치-및-실행)
5. [핵심 기능](#5-핵심-기능)
6. [코드 구조](#6-코드-구조)
7. [데이터 관리](#7-데이터-관리)
8. [트러블슈팅](#8-트러블슈팅)
9. [개발 가이드](#9-개발-가이드)

---

## 1. 프로젝트 개요

### 1.1 프로젝트 목적
한국 안전 자격증 시험(산업안전기사, 방재기사, 건설안전기사, 방재안전직)에 대한 **LLM 벤치마크 결과를 다차원적으로 분석**하는 웹 기반 시각화 도구

### 1.2 주요 특징
- ✅ **8개 이상의 분석 탭**: 전체 요약, 모델 비교, 응답시간, 법령/비법령, 과목별, 연도별, 오답 분석, 난이도 분석 등
- ✅ **앙상블 모델 기능**: 여러 모델의 결과를 조합하여 새로운 앙상블 모델 생성
- ✅ **자동 모델 인식**: 파일명에서 모델명을 자동으로 파싱하여 코드 수정 없이 새 모델 추가 가능
- ✅ **다국어 지원**: 한국어/영어
- ✅ **포괄적인 내보내기**: 모든 표를 Excel/CSV/TSV로 다운로드 가능
- ✅ **GitHub Releases 데이터 관리**: 대용량 벤치마크 데이터를 Release로 관리

### 1.3 기술 스택
```
프론트엔드: Streamlit (Python web framework)
데이터 처리: Pandas, NumPy
시각화: Plotly (interactive charts)
통계: SciPy
파일 처리: openpyxl (Excel), io (in-memory files)
배포: Streamlit Cloud + GitHub
```

---

## 2. 시스템 아키텍처

### 2.1 전체 구조
```
┌─────────────────────────────────────────────────────┐
│              Streamlit Web Interface                │
│  (llm_benchmark_visualizer.py)                      │
└──────────────────┬──────────────────────────────────┘
                   │
       ┌───────────┴───────────┐
       │                       │
┌──────▼──────┐        ┌──────▼──────┐
│  Testsets   │        │   Results   │
│  (원본 문제) │        │ (모델 평가)  │
└─────────────┘        └─────────────┘
       │                       │
       └───────────┬───────────┘
                   │
           ┌───────▼───────┐
           │  Data Analysis │
           │  & Filtering   │
           └───────┬────────┘
                   │
       ┌───────────┴───────────┐
       │                       │
┌──────▼──────┐        ┌──────▼──────┐
│   Plotly    │        │  Export to  │
│   Charts    │        │ Excel/CSV   │
└─────────────┘        └─────────────┘
```

### 2.2 데이터 흐름
```
1. GitHub Release → download_data_from_github()
2. Load → load_data(data_dir)
3. Filter → Sidebar filters
4. Analyze → Per-tab analysis functions
5. Visualize → Plotly charts
6. Export → Excel/CSV/TSV downloads
```

---

## 3. 데이터 구조

### 3.1 파일 명명 규칙

#### ✅ **Testset 파일** (원본 문제 데이터)
```
형식: testset_{테스트명}.csv

예시:
testset_산업안전기사.csv
testset_방재기사.csv
testset_건설안전기사.csv
testset_방재안전직.csv
```

#### ✅ **Result 파일** (벤치마크 결과)
```
형식: {모델명}_{상세도}_{프롬프팅방식}_{테스트명}.csv

구성요소:
- 모델명: gpt-4o, claude-3-5-sonnet, llama-3-3-70b 등
- 상세도: detailed 또는 summary
- 프롬프팅방식: noprompting, few-shot, cot 등
- 테스트명: 산업안전기사, 방재기사 등

예시:
gpt-4o_detailed_noprompting_산업안전기사.csv
claude-3-5-sonnet_summary_few-shot_방재기사.csv
llama-3-3-70b_detailed_cot_건설안전기사.csv
```

### 3.2 Testset CSV 구조
```csv
Question,Answer,law,Subject,Year,Session,Number,Choice_1,Choice_2,Choice_3,Choice_4,image_path
문제 내용,2,O,안전관리론,2024,1,1,선택지1,선택지2,선택지3,선택지4,images/2024_1_1.png
```

**필수 컬럼:**
- `Question`: 문제 내용
- `Answer`: 정답 (1, 2, 3, 4)
- `law`: 법령 문제 여부 ('O' 또는 'X')
- `Subject`: 과목명
- `Year`: 출제 연도
- `Session`: 회차 (1, 2, 3 등)
- `Number`: 문제 번호

**선택 컬럼:**
- `Choice_1`, `Choice_2`, `Choice_3`, `Choice_4`: 선택지
- `image_path`: 이미지 경로

### 3.3 Result CSV 구조
```csv
Question,Answer,예측답,정답여부,응답시간,입력토큰,출력토큰,총토큰,law,Subject,Year,Session
문제내용,2,2,True,1.23,450,120,570,O,안전관리론,2024,1
```

**필수 컬럼:**
- `Question`: 문제 내용 (Testset과 매칭용)
- `Answer`: 정답
- `예측답`: 모델이 예측한 답 (1, 2, 3, 4 또는 nan)
- `정답여부`: True/False

**선택 컬럼:**
- `응답시간`: 응답 시간 (초)
- `입력토큰`, `출력토큰`, `총토큰`: 토큰 사용량
- Testset에서 복사된 메타데이터: `law`, `Subject`, `Year`, `Session` 등

---

## 4. 설치 및 실행

### 4.1 환경 요구사항
```bash
Python >= 3.8
```

### 4.2 로컬 설치
```bash
# 1. 저장소 클론
git clone https://github.com/kjs9964/benchmark_visualizer.git
cd benchmark_visualizer

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 실행
streamlit run llm_benchmark_visualizer.py

# 또는
chmod +x run.sh
./run.sh
```

### 4.3 Streamlit Cloud 배포
```
1. GitHub 저장소 생성
2. Streamlit Cloud (https://streamlit.io/cloud) 로그인
3. "New app" → GitHub 저장소 선택
4. Main file: llm_benchmark_visualizer.py
5. Deploy!
```

### 4.4 데이터 자동 다운로드
- 앱 시작 시 GitHub Releases에서 `data.zip` 자동 다운로드
- Release 태그: `v2.2.0`
- URL: `https://github.com/kjs9964/benchmark_visualizer/releases/download/v2.2.0/data.zip`
- 캐시: 24시간 (86400초)

---

## 5. 핵심 기능

### 5.1 필터링 옵션 (사이드바)
```python
✅ 테스트명 (multiselect)
✅ 모델 (multiselect)
✅ 상세도 (multiselect): detailed, summary
✅ 프롬프팅 방식 (multiselect): no-prompting, few-shot, cot
✅ 세션 (multiselect)
✅ 문제 유형 (radio): 전체, 이미지 문제, 텍스트만
✅ 연도 (multiselect)
✅ 법령 구분 (radio): 전체, 법령, 비법령
✅ 언어 (selectbox): 한국어, English
✅ 폰트 크기 (slider): 0.8 ~ 1.5
✅ 차트 텍스트 크기 (slider): 0.7 ~ 1.8
```

### 5.2 분석 탭 (Tabs)

#### **Tab 1: 📊 전체 요약**
- 테스트셋 정보 (총 문제 수, 평가 모델 수, 총 평가 횟수)
- 모델 평균 성능 (평균 정확도, 정답/오답 수)
- 법령/비법령 전체 통계 (파이 차트)
- 모델별 성능 지표 테이블 (정확도, 정답/오답 수, 법령/비법령 정확도)
- 모델별 성능 비교 차트 (막대 그래프)
- 전체 테스트 비교 (테스트별 평균 정확도)
- 모델별 테스트셋 정답도 히트맵

#### **Tab 2: 🔍 모델별 비교**
- 모델별 법령/비법령 성능 비교 (그룹 막대 그래프)
- 모델별 법령/비법령 세부 통계 테이블
- 모델별 과목별 성능 히트맵
- 모델별 과목별 평균 정확도 표
- 모델별 연도별 성능 트렌드 (꺾은선 그래프)
- 모델 순위 비교 (정확도, 응답시간, 토큰 효율)

#### **Tab 3: ⏱️ 응답시간 분석**
- 모델별 평균 응답시간 비교 (막대 그래프)
- 응답시간 분포 (박스 플롯)
- 정확도 vs 응답시간 산점도
- 응답시간 통계 테이블 (평균, 중앙값, 최소, 최대, 표준편차)

#### **Tab 4: ⚖️ 법령/비법령 분석**
- 법령/비법령 전체 통계 (파이 차트)
- 모델별 법령/비법령 성능 비교 (그룹 막대 그래프)
- 법령/비법령 정답률 차이 분석 (차이값 막대 그래프)
- 법령/비법령 세부 통계 테이블

#### **Tab 5: 📚 과목별 분석**
- 과목별 전체 성능 (평균 정확도 막대 그래프)
- 모델별 과목별 성능 히트맵
- 과목별 모델 순위 테이블
- 과목별 상세 통계 (문제 수, 평균 정확도)

#### **Tab 6: 📅 연도별 분석**
- 연도별 전체 성능 트렌드 (꺾은선 그래프)
- 모델별 연도별 성능 비교 (꺾은선 그래프)
- 연도별 평균 정확도 테이블
- 연도별 난이도 변화 분석

#### **Tab 7: ❌ 오답 분석**
- **공통 오답 핵심 지표**: 여러 모델이 일관되게 같은 오답을 선택한 문제 식별
- 오답률 50% 이상 문제만 분석 (의미 있는 패턴)
- 일관성 계산: (가장 많은 오답 횟수) / (전체 오답 횟수)
- 공통 오답 상세 테이블 (문제 정보, 정답, 공통 오답, 일관성)
- 모델별 공통 오답 비율
- 오답 패턴 시각화 (히트맵)

#### **Tab 8: 📈 난이도 분석**
- 문제별 난이도 점수 계산 (정답률 기반)
- 난이도 구간별 문제 분포 (매우 어려움, 어려움, 보통, 쉬움, 매우 쉬움)
- 모델별 난이도 구간별 정확도 (꺾은선 그래프)
- 난이도 vs 정확도 산점도
- 난이도 구간별 상세 테이블

#### **Tab 9: 💰 토큰 및 비용 분석**
- 모델별 토큰 사용량 통계 (입력, 출력, 총 토큰)
- 모델별 평균 토큰 사용량 비교 (막대 그래프)
- 비용 효율성 분석 (정확도 대비 토큰 사용량)
- 토큰-정확도 산점도

#### **Tab 10: 📋 테스트셋 통계**
- 전체 테스트셋 통계 (총 문제 수, 법령/비법령 문제 수)
- 테스트별 개별 통계 (문제 수, 법령 비율)
- 과목별, 연도별, 세션별 문제 수

#### **Tab 11: 📑 추가 분석**
- 테스트셋별 평균 정답률 표
- 모델-테스트 조합 성능 매트릭스
- 상세 통계 (평균, 표준편차, 최소, 최대)

### 5.3 앙상블 모델 기능
```python
# 앙상블 생성 방법
1. 사이드바 → "➕ 앙상블 생성" 펼치기
2. 앙상블 이름 입력 (예: "GPT 앙상블")
3. 모델 선택 (최소 2개)
4. 앙상블 방법 선택:
   - 다수결 투표 (Majority Voting): 가장 많이 선택된 답 선택
   - 가중 투표 (Weighted Voting): 모델 정확도 기반 가중치
5. "✅ 앙상블 추가" 클릭

# 앙상블 작동 원리
- 다수결: 가장 많은 모델이 선택한 답을 최종 답으로 선택
- 가중 투표: (모델 정확도) × (선택 횟수)가 가장 높은 답 선택

# 앙상블 삭제
- 현재 앙상블 목록에서 "🗑️" 버튼 클릭
```

### 5.4 데이터 내보내기
```python
# 모든 표에 대해 3가지 내보내기 옵션 제공
📥 Excel 다운로드: .xlsx 파일
📄 CSV 다운로드: .csv 파일 (범용)
📋 TSV 다운로드: .tsv 파일 (Excel에서 헤더 포함 복사용)

# Excel 복사 가이드
1. 표 좌측 상단 📋 아이콘 클릭 → 헤더 포함 전체 복사
2. Excel에 붙여넣기
```

---

## 6. 코드 구조

### 6.1 파일 목록
```
benchmark_visualizer/
├── llm_benchmark_visualizer.py  # 메인 앱
├── validate_data.py             # 데이터 검증 스크립트
├── requirements.txt             # Python 의존성
├── run.sh                       # 실행 스크립트 (Bash)
├── .gitignore                   # Git 무시 목록
├── .devcontainer/
│   └── devcontainer.json        # VSCode Dev Container 설정
└── README.md                    # (선택) 프로젝트 설명
```

### 6.2 주요 함수 구조

#### **llm_benchmark_visualizer.py**

```python
# ========== 데이터 로드 ==========
@st.cache_data(ttl=86400)
def download_data_from_github():
    """GitHub Release에서 data.zip 다운로드 및 압축 해제"""

@st.cache_data
def load_data(data_dir):
    """
    - testset 파일 로드 → testsets 딕셔너리
    - result 파일 로드 → results_df DataFrame
    - 파일명에서 모델명, 상세도, 프롬프팅, 테스트명 자동 파싱
    """

# ========== 앙상블 모델 ==========
def create_ensemble_model(results_df, ensemble_name, models, method):
    """
    여러 모델의 예측을 결합하여 앙상블 모델 생성
    - method: 'majority' (다수결) or 'weighted' (가중 투표)
    """

# ========== 유틸리티 ==========
def safe_convert_to_int(value):
    """쉼표 구분자가 있는 문자열을 정수로 안전하게 변환"""

def format_question_id(row):
    """행 정보를 기반으로 문제 식별자 생성 (예: 산업안전기사 / 2024 / S1 / 안전관리론 / Q1)"""

def get_testset_statistics(testsets, test_name, lang='ko'):
    """테스트셋의 기초 통계 반환 (총 문제 수, 법령/비법령, 과목별, 연도별)"""

# ========== 분석 함수 ==========
def calculate_model_release_date(model_name):
    """모델명으로부터 출시 시기 추정"""

def calculate_model_parameters(model_name):
    """모델명으로부터 파라미터 수 추정 (억 단위)"""

def create_testset_accuracy_table(filtered_df, testsets, lang='ko'):
    """테스트셋별 평균 정답률 표 생성"""

# ========== 내보내기 ==========
def create_download_button(df, filename):
    """Excel (.xlsx) 다운로드 버튼 생성"""

def create_csv_download_button(df, filename):
    """CSV 다운로드 버튼 생성"""

def create_copy_button(df, button_text, key_suffix):
    """TSV 다운로드 버튼 생성 (Excel 헤더 포함 복사용)"""

def display_table_with_download(df, title, excel_filename, lang='ko'):
    """표 + 3개 다운로드 버튼 (Excel, CSV, TSV) 함께 표시"""

# ========== 스타일링 ==========
def apply_custom_css(font_size=1.0):
    """전역 CSS 스타일 적용 (폰트 크기 조절)"""

def set_plotly_font_size(scale=1.0):
    """Plotly 차트 폰트 크기 설정"""

# ========== 메인 ==========
def main():
    """메인 앱 로직"""
    # 1. 데이터 다운로드 및 로드
    # 2. 사이드바 필터 구성
    # 3. 앙상블 모델 관리
    # 4. 탭별 분석 및 시각화
    # 5. 내보내기 기능

if __name__ == "__main__":
    main()
```

#### **validate_data.py**

```python
def validate_data(data_dir="/mnt/project"):
    """
    데이터 검증 및 기본 통계 출력
    - Testset 파일 확인 (문제 수, 법령 비율, 과목 수)
    - Result 파일 확인 (모델별 파일 수, 정답 수, 정확도)
    """
```

### 6.3 데이터 파싱 로직

**파일명 파싱 예시:**
```python
# 입력: gpt-4o_detailed_noprompting_산업안전기사.csv

# 파싱 결과:
model = "GPT-4o"          # 모델명 정규화 및 포맷팅
detail_type = "detailed"  # 상세도
prompt_type = "no-prompting"  # 프롬프팅 방식 정규화
test_name = "산업안전기사"  # 테스트명 (testset 목록에서 자동 감지)

# 모델명 정규화 규칙:
# - 첫 글자 대문자
# - 연속된 한 자리 숫자 → 버전 번호 (3, 5 → 3.5)
# - 특수 규칙: "4o" → "4o", "mini" 뒤에 숫자 → "Mini-숫자"
```

---

## 7. 데이터 관리

### 7.1 GitHub Releases 활용

**왜 GitHub Releases를 사용하나?**
- ✅ 대용량 CSV 파일을 Git 저장소에 포함하지 않음 (저장소 크기 절감)
- ✅ 자동 다운로드 및 캐싱 (Streamlit Cloud에서도 작동)
- ✅ 버전 관리 용이 (태그로 데이터 버전 관리)

**데이터 업로드 방법:**
```bash
# 1. data 폴더에 모든 CSV 파일 배치
data/
├── testset_산업안전기사.csv
├── testset_방재기사.csv
├── gpt-4o_detailed_noprompting_산업안전기사.csv
├── gpt-4o_summary_noprompting_산업안전기사.csv
└── ... (기타 result 파일들)

# 2. data.zip 생성
zip -r data.zip data/

# 3. GitHub Release 생성
# - 저장소 → Releases → "Create a new release"
# - Tag version: v2.2.0 (예시)
# - Release title: Data Release v2.2.0
# - Upload data.zip

# 4. llm_benchmark_visualizer.py 수정
# - download_data_from_github() 함수의 tag 변수 업데이트
tag = "v2.2.0"  # 새 태그로 변경
```

### 7.2 새 벤치마크 데이터 추가

**Step 1: 새 testset 추가**
```bash
# 새 테스트셋 생성 (예: 전기안전기사)
data/testset_전기안전기사.csv

# 필수 컬럼 확인:
Question,Answer,law,Subject,Year,Session,Number,...
```

**Step 2: 새 result 파일 추가**
```bash
# 명명 규칙 준수
{모델명}_{상세도}_{프롬프팅}_{테스트명}.csv

# 예시:
gpt-4o_detailed_noprompting_전기안전기사.csv
gpt-4o_summary_few-shot_전기안전기사.csv
```

**Step 3: 데이터 업로드**
```bash
# data.zip 재생성 후 GitHub Release 업데이트
zip -r data.zip data/
# → GitHub Release에 업로드
```

**Step 4: 코드 수정 없음!**
- ✅ testset 파일명에서 자동으로 테스트명 추출
- ✅ result 파일명에서 자동으로 메타데이터 파싱
- ✅ 필터 옵션 자동 업데이트

### 7.3 새 모델 추가

**자동 인식 방식:**
```python
# 코드 수정 없이 새 모델 자동 추가!
# 파일명만 규칙에 맞으면 자동 파싱됨

# 예시: Gemini-Pro 추가
gemini-pro_detailed_noprompting_산업안전기사.csv
→ 모델명 자동 파싱: "Gemini-Pro"

# 예시: Llama-4-405b 추가
llama-4-405b_summary_cot_방재기사.csv
→ 모델명 자동 파싱: "Llama-4-405b"
```

**수동 정규화가 필요한 경우:**
```python
# llm_benchmark_visualizer.py → load_data() 함수 내
# format_model_name() 함수에서 특수 규칙 추가

# 예시: "openai-o1" → "OpenAI-o1"로 변환하고 싶은 경우
def format_model_name(model_normalized):
    # 기존 코드...
    
    # 커스텀 규칙 추가
    if model_normalized.startswith('openai_o1'):
        return 'OpenAI-o1'
    
    # 나머지 자동 파싱...
```

---

## 8. 트러블슈팅

### 8.1 자주 발생하는 문제

#### **문제 1: 데이터가 로드되지 않음**
```
증상: "No data files found" 경고

해결:
1. GitHub Release 확인
   - URL: https://github.com/kjs9964/benchmark_visualizer/releases
   - data.zip 파일이 정상적으로 업로드되었는지 확인

2. 로컬에서 테스트
   python validate_data.py ./data

3. 파일명 규칙 확인
   - testset_{테스트명}.csv ✅
   - {모델}_{상세도}_{프롬프팅}_{테스트명}.csv ✅
```

#### **문제 2: 앙상블 모델이 생성되지 않음**
```
증상: 앙상블 추가 후 결과가 표시되지 않음

해결:
1. 최소 2개 모델 선택 확인
2. 선택한 모델들이 같은 문제를 평가했는지 확인
   - Question 컬럼 값이 일치해야 함
3. 앙상블 방법 확인
   - 다수결: 최소 2개 이상 같은 답
   - 가중: 정확도 계산 가능해야 함
```

#### **문제 3: 차트가 제대로 표시되지 않음**
```
증상: 빈 차트 또는 오류 메시지

해결:
1. 데이터 필터링 확인
   - 필터가 너무 제한적이지 않은지 확인
   - 최소 1개 이상의 데이터 포인트 필요

2. 필수 컬럼 확인
   - 각 분석 탭에서 요구하는 컬럼이 있는지 확인
   - 예: 응답시간 분석 → '응답시간' 컬럼 필요

3. 브라우저 콘솔 확인
   - F12 → Console 탭에서 에러 메시지 확인
```

#### **문제 4: CSV 인코딩 오류**
```
증상: UnicodeDecodeError

해결:
1. CSV 파일을 UTF-8로 저장
   - Excel: "CSV UTF-8 (쉼표로 분리)" 형식으로 저장

2. 코드에서 자동으로 cp949도 시도함
   try:
       df = pd.read_csv(file, encoding='utf-8')
   except:
       df = pd.read_csv(file, encoding='cp949')
```

### 8.2 성능 최적화

#### **대용량 데이터 처리**
```python
# 1. 캐싱 활용
@st.cache_data  # 데이터 로드 결과 캐싱
def load_data(data_dir):
    ...

# 2. 필터링 최적화
# 불필요한 컬럼 제거
filtered_df = results_df[required_columns].copy()

# 3. 차트 렌더링 최적화
# Plotly의 scattergl 사용 (대량 데이터 포인트)
fig = px.scatter_gl(...)  # 대신 scatter 사용
```

#### **메모리 사용량 줄이기**
```python
# 1. 데이터 타입 최적화
df['Year'] = df['Year'].astype('int16')  # int64 대신
df['정답여부'] = df['정답여부'].astype('bool')  # object 대신

# 2. 불필요한 복사 제거
# 나쁜 예:
temp = df.copy()  # 메모리 2배 사용

# 좋은 예:
temp = df  # 참조만 복사
```

---

## 9. 개발 가이드

### 9.1 새 분석 탭 추가하기

**Step 1: 탭 목록에 추가**
```python
# main() 함수 내
tabs = st.tabs([
    f"📊 {t['overview']}",
    f"🔍 {t['model_comparison']}",
    # ... 기존 탭들 ...
    "🆕 새 분석",  # ← 여기에 추가
])
```

**Step 2: 탭 내용 구현**
```python
# 탭 12: 새 분석
with tabs[11]:  # 인덱스는 0부터 시작
    st.header("🆕 새 분석")
    
    # 데이터 처리
    analysis_df = filtered_df.groupby('모델')['정답여부'].mean()
    
    # 시각화
    fig = px.bar(
        x=analysis_df.index,
        y=analysis_df.values,
        title="새 분석 차트"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 표 표시 및 다운로드
    display_table_with_download(
        analysis_df.to_frame(),
        "새 분석 표",
        "new_analysis.xlsx",
        lang
    )
```

**Step 3: 다국어 지원 추가**
```python
# LANGUAGES 딕셔너리에 추가
LANGUAGES = {
    'ko': {
        # ... 기존 항목들 ...
        'new_analysis': '새 분석',
        'new_analysis_title': '새 분석 차트',
    },
    'en': {
        # ... 기존 항목들 ...
        'new_analysis': 'New Analysis',
        'new_analysis_title': 'New Analysis Chart',
    }
}
```

### 9.2 새 필터 추가하기

```python
# 사이드바 필터 섹션에 추가
st.sidebar.markdown("---")

# 예: 난이도 필터
difficulty_options = ['전체', '쉬움', '보통', '어려움']
selected_difficulty = st.sidebar.selectbox(
    "난이도",
    options=difficulty_options
)

# 필터 적용
if selected_difficulty != '전체':
    # 난이도 컬럼이 있다고 가정
    filtered_df = filtered_df[filtered_df['difficulty'] == selected_difficulty]
```

### 9.3 새 차트 유형 추가하기

```python
# Plotly 차트 종류
import plotly.express as px
import plotly.graph_objects as go

# 1. 막대 그래프
fig = px.bar(df, x='category', y='value')

# 2. 꺾은선 그래프
fig = px.line(df, x='year', y='accuracy', color='model')

# 3. 산점도
fig = px.scatter(df, x='time', y='accuracy', color='model')

# 4. 박스 플롯
fig = px.box(df, x='model', y='response_time')

# 5. 히트맵
fig = px.imshow(pivot_table, text_auto=True)

# 6. 파이 차트
fig = px.pie(df, values='count', names='category')

# 차트 커스터마이징
fig.update_layout(
    title="차트 제목",
    xaxis_title="X축",
    yaxis_title="Y축",
    height=500
)

# 표시
st.plotly_chart(fig, use_container_width=True)
```

### 9.4 코드 스타일 가이드

```python
# ✅ 좋은 예
def calculate_accuracy(df: pd.DataFrame) -> float:
    """
    데이터프레임의 정확도 계산
    
    Args:
        df: 정답여부 컬럼이 있는 DataFrame
        
    Returns:
        평균 정확도 (0.0 ~ 1.0)
    """
    if '정답여부' not in df.columns:
        return 0.0
    return df['정답여부'].mean()

# ❌ 나쁜 예
def calc(d):
    return d['정답여부'].mean()  # 문서화 없음, 타입 힌트 없음

# 변수명 규칙
good_name = "filtered_df"  # ✅ 명확한 의미
bad_name = "df2"  # ❌ 의미 불명확

# 주석
# ✅ 왜 이렇게 했는지 설명
# 백업: testsets에 없으면 고유 문제 수로 추정
actual_problems = testset_df['Question'].nunique()

# ❌ 코드 그대로 반복
# 문제 수 계산
actual_problems = len(df)
```

### 9.5 테스트 방법

```bash
# 1. 데이터 검증
python validate_data.py ./data

# 2. 로컬 실행
streamlit run llm_benchmark_visualizer.py

# 3. 체크리스트
□ 모든 필터가 작동하는가?
□ 모든 탭이 에러 없이 표시되는가?
□ 다운로드 버튼이 작동하는가?
□ 앙상블 모델 생성/삭제가 작동하는가?
□ 다국어 전환이 작동하는가?
□ 폰트 크기 조절이 작동하는가?
```

---

## 10. 참고 자료

### 10.1 기술 문서
- **Streamlit**: https://docs.streamlit.io/
- **Plotly**: https://plotly.com/python/
- **Pandas**: https://pandas.pydata.org/docs/
- **SciPy**: https://docs.scipy.org/doc/scipy/

### 10.2 프로젝트 링크
- **GitHub 저장소**: https://github.com/kjs9964/benchmark_visualizer
- **Streamlit Cloud**: (배포 후 링크 추가)

### 10.3 연락처
- **담당자**: 지성
- **이메일**: (필요시 추가)
- **GitHub**: @kjs9964

---

## 11. 체크리스트

### 11.1 인수인계 체크리스트

#### **프로젝트 이해**
- [ ] 프로젝트 목적 및 주요 기능 이해
- [ ] 데이터 구조 및 파일 명명 규칙 이해
- [ ] 시스템 아키텍처 이해

#### **환경 설정**
- [ ] Python 환경 구축 완료
- [ ] 의존성 설치 완료 (`pip install -r requirements.txt`)
- [ ] 로컬에서 앱 실행 성공

#### **데이터 관리**
- [ ] GitHub Releases 접근 권한 확인
- [ ] 데이터 업로드 프로세스 이해
- [ ] 새 테스트셋/모델 추가 방법 숙지

#### **기능 테스트**
- [ ] 모든 필터 작동 확인
- [ ] 모든 분석 탭 작동 확인
- [ ] 앙상블 모델 생성/삭제 확인
- [ ] 데이터 내보내기 확인
- [ ] 다국어 전환 확인

#### **개발 역량**
- [ ] 새 분석 탭 추가 방법 이해
- [ ] 새 필터 추가 방법 이해
- [ ] 새 차트 추가 방법 이해
- [ ] 코드 스타일 가이드 숙지

#### **트러블슈팅**
- [ ] 자주 발생하는 문제 및 해결 방법 숙지
- [ ] 성능 최적화 방법 이해

---

## 12. 버전 히스토리

### v2.2.0 (현재)
- ✅ GitHub Releases 데이터 관리
- ✅ 앙상블 모델 기능
- ✅ 공통 오답 분석 강화
- ✅ 포괄적인 내보내기 기능
- ✅ 다국어 지원 (한국어/영어)

### v2.1.0
- ✅ 난이도 분석 추가
- ✅ 토큰 및 비용 분석 추가
- ✅ 차트 텍스트 크기 조절 기능

### v2.0.0
- ✅ 8개 분석 탭 구현
- ✅ 자동 모델 인식
- ✅ 테스트셋 통계 탭

### v1.0.0
- ✅ 기본 시각화 기능
- ✅ 모델별 비교

---

**📝 마지막 업데이트: 2024-12-08**

**✨ 프로젝트 인수인계를 환영합니다! 궁금한 점이 있으면 언제든지 문의해주세요.**
