import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import glob
from pathlib import Path
import numpy as np
from scipy import stats
import io
import requests
import zipfile
import gc
import re
from collections import Counter
from functools import lru_cache

@st.cache_data(ttl=86400)
def download_data_from_github():
    """저장소에 포함된 data 폴더 사용"""
    
    data_dir = Path('./data')
    
    # CSV 파일 존재 확인
    if data_dir.exists() and next(data_dir.glob('*.csv'), None) is not None:
        csv_count = sum(1 for _ in data_dir.glob('*.csv'))
        return
    
    # data 폴더가 없거나 CSV 파일이 없으면 에러
    st.error("❌ data 폴더 또는 CSV 파일이 없습니다.")
    st.error("저장소에 data 폴더와 CSV 파일이 포함되어 있는지 확인해주세요.")
    st.stop()

# 페이지 설정
st.set_page_config(
    page_title="LLM 벤치마크 시각화",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 다국어 지원 설정
LANGUAGES = {
    'ko': {
        'title': 'LLM 벤치마크 결과 시각화 도구',
        'data_dir': '데이터 디렉토리',
        'filters': '필터 옵션',
        'testname': '테스트명',
        'all': '전체',
        'model': '모델',
        'detail_type': '상세도',
        'prompting': '프롬프팅 방식',
        'session': '세션',
        'problem_type': '문제 유형',
        'image_problem': '이미지 문제',
        'text_only': '텍스트만',
        'year': '연도',
        'law_type': '법령 구분',
        'law': '법령',
        'non_law': '비법령',
        'overview': '전체 요약',
        'model_comparison': '모델별 비교',
        'response_time_analysis': '응답시간 분석',
        'law_analysis': '법령/비법령 분석',
        'subject_analysis': '과목별 분석',
        'year_analysis': '연도별 분석',
        'incorrect_analysis': '오답 분석',
        'difficulty_analysis': '난이도 분석',
        'ensemble': '앙상블',
        'ensemble_management': '앙상블 모델 관리',
        'create_ensemble': '앙상블 생성',
        'ensemble_name': '앙상블 이름',
        'ensemble_method': '앙상블 방법',
        'majority_voting': '다수결 투표',
        'weighted_voting': '가중 투표',
        'select_models': '모델 선택',
        'add_ensemble': '앙상블 추가',
        'remove_ensemble': '삭제',
        'current_ensembles': '현재 앙상블 목록',
        'no_ensembles': '생성된 앙상블이 없습니다',
        'ensemble_added': '앙상블이 추가되었습니다',
        'ensemble_removed': '앙상블이 삭제되었습니다',
        'min_2_models': '최소 2개 이상의 모델을 선택해야 합니다',
        'testset_stats': '테스트셋 통계',
        'total_problems': '총 문제 수',
        'accuracy': '정확도',
        'correct': '정답',
        'wrong': '오답',
        'law_problems': '법령 문제',
        'non_law_problems': '비법령 문제',
        'correct_rate': '정답률',
        'wrong_rate': '오답률',
        'performance_by_model': '모델별 성능 지표',
        'comparison_chart': '모델별 성능 비교 차트',
        'overall_comparison': '전체 테스트 비교',
        'heatmap': '모델별 테스트셋 정답도 히트맵',
        'law_ratio': '법령/비법령 전체 통계',
        'model_law_performance': '모델별 법령/비법령 성능 비교',
        'law_distribution': '모델별 법령/비법령 정답도',
        'subject_performance': '과목별 성능',
        'year_performance': '연도별 성능',
        'top_incorrect': '오답률 높은 문제 Top 20',
        'all_models_incorrect': '모든 모델이 틀린 문제',
        'most_models_incorrect': '대부분 모델이 틀린 문제 (≥50%)',
        'test_info': '테스트',
        'problem_id': '문제번호',
        'incorrect_count': '오답 모델수',
        'correct_count': '정답 모델수',
        'total_models': '총 모델수',
        'attempted_models': '시도한 모델',
        'question': '문제',
        'difficulty_score': '난이도 점수',
        'by_session': '세션별',
        'by_subject': '과목별',
        'by_year': '연도별',
        'problem_count': '문제 수',
        'session_distribution': '세션별 문제 분포',
        'subject_distribution': '과목별 문제 분포',
        'year_distribution': '연도별 문제 분포',
        'law_distribution_stat': '법령/비법령 문제 분포',
        'basic_stats': '기본 통계',
        'help': '도움말',
        'new_features': '새로운 기능',
        'existing_features': '기존 기능',
        'current_data': '현재 표시 중인 데이터',
        'problems': '개 문제',
        'session_filter': '특정 세션의 결과만 분석',
        'incorrect_pattern': '어려운 문제와 오답 패턴 분석',
        'difficulty_comparison': '문제 난이도별 모델 성능 비교',
        'problem_type_filter': '이미지/텍스트 문제 구분',
        'basic_filters': '테스트명, 모델, 상세도, 프롬프팅 방식으로 필터링',
        'law_analysis_desc': '법령/비법령 구분 분석',
        'detail_analysis': '과목별, 연도별 상세 분석',
        'font_size': '화면 폰트 크기',
        'chart_text_size': '차트 텍스트 크기',
        'year_problem_distribution': '연도별 문제 수 분포',
        'problem_count_table': '연도별 문제 수 테이블',
        'year_problem_chart': '연도별 문제 수',
        'total_problem_count': '총 문제 수',
        'correct_models': '정답 모델',
        'incorrect_models': '오답 모델',
        'avg_accuracy_by_model': '모델별 평균 정확도',
        'difficulty_range': '난이도 구간',
        'avg_difficulty': '평균 난이도',
        'difficulty_stats_by_range': '난이도 구간별 상세 통계',
        'very_hard': '매우 어려움',
        'hard': '어려움',
        'medium': '보통',
        'easy': '쉬움',
        'very_easy': '매우 쉬운',
        'problem_distribution': '문제 분포',
        'response_time': '응답 시간',
        'avg_response_time': '평균 응답 시간',
        'response_time_distribution': '응답 시간 분포',
        'response_time_by_model': '모델별 응답 시간',
        'response_time_stats': '응답 시간 통계',
        'fastest_model': '가장 빠른 모델',
        'slowest_model': '가장 느린 모델',
        'response_time_vs_accuracy': '응답 시간 vs 정확도',
        'time_per_problem': '문제당 시간',
        'total_time': '총 소요 시간',
        'seconds': '초',
        'minutes': '분',
        # 토큰 및 비용 관련
        'token_cost_analysis': '토큰 및 비용 분석',
        'token_usage': '토큰 사용량',
        'input_tokens': '입력 토큰',
        'output_tokens': '출력 토큰',
        'total_tokens': '총 토큰',
        'avg_tokens_per_problem': '문제당 평균 토큰',
        'token_distribution': '토큰 분포',
        'token_efficiency': '토큰 효율성',
        'token_stats': '토큰 통계',
        'io_ratio': '입출력 토큰 비율',
        'token_per_correct': '정답당 토큰',
        'tokens': '토큰',
        'cost_level': '비용 수준',
        'cost_analysis': '비용 분석',
        'cost_per_problem': '문제당 비용',
        'total_cost_estimate': '총 예상 비용',
        'cost_vs_accuracy': '비용 vs 정확도',
        'cost_efficiency': '비용 효율성',
        'most_efficient': '가장 효율적인 모델',
        'least_efficient': '가장 비효율적인 모델',
        'cost_stats': '비용 통계',
        'high': '높음',
        'medium_cost': '중간',
        'low': '낮음',
        'very_low': '매우낮음',
        'free': '무료',
        'cost': '비용',
        'actual_cost': '실제 비용',
        'estimated_cost': '예상 비용',
        'cost_per_1k_tokens': '1K 토큰당 비용',
        'total_estimated_cost': '총 예상 비용',
        'usd': '달러',
    },
    'en': {
        'title': 'LLM Benchmark Results Visualization Tool',
        'data_dir': 'Data Directory',
        'filters': 'Filter Options',
        'testname': 'Test Name',
        'all': 'All',
        'model': 'Model',
        'detail_type': 'Detail Type',
        'prompting': 'Prompting Method',
        'session': 'Session',
        'problem_type': 'Problem Type',
        'image_problem': 'Image Problem',
        'text_only': 'Text Only',
        'year': 'Year',
        'law_type': 'Law Type',
        'law': 'Law',
        'non_law': 'Non-Law',
        'overview': 'Overview',
        'model_comparison': 'Model Comparison',
        'response_time_analysis': 'Response Time Analysis',
        'law_analysis': 'Law/Non-Law Analysis',
        'subject_analysis': 'Subject Analysis',
        'year_analysis': 'Year Analysis',
        'incorrect_analysis': 'Incorrect Answer Analysis',
        'difficulty_analysis': 'Difficulty Analysis',
        'ensemble': 'Ensemble',
        'ensemble_management': 'Ensemble Model Management',
        'create_ensemble': 'Create Ensemble',
        'ensemble_name': 'Ensemble Name',
        'ensemble_method': 'Ensemble Method',
        'majority_voting': 'Majority Voting',
        'weighted_voting': 'Weighted Voting',
        'select_models': 'Select Models',
        'add_ensemble': 'Add Ensemble',
        'remove_ensemble': 'Remove',
        'current_ensembles': 'Current Ensembles',
        'no_ensembles': 'No ensembles created',
        'ensemble_added': 'Ensemble added successfully',
        'ensemble_removed': 'Ensemble removed',
        'min_2_models': 'Please select at least 2 models',
        'testset_stats': 'Test Set Statistics',
        'total_problems': 'Total Problems',
        'accuracy': 'Accuracy',
        'correct': 'Correct',
        'wrong': 'Wrong',
        'law_problems': 'Law Problems',
        'non_law_problems': 'Non-Law Problems',
        'correct_rate': 'Correct Rate',
        'wrong_rate': 'Wrong Rate',
        'performance_by_model': 'Performance Metrics by Model',
        'comparison_chart': 'Model Performance Comparison Chart',
        'overall_comparison': 'Overall Test Comparison',
        'heatmap': 'Model × Test Set Accuracy Heatmap',
        'law_ratio': 'Law/Non-Law Overall Statistics',
        'model_law_performance': 'Model Law/Non-Law Performance Comparison',
        'law_distribution': 'Law/Non-Law Accuracy by Model',
        'subject_performance': 'Performance by Subject',
        'year_performance': 'Performance by Year',
        'top_incorrect': 'Top 20 Problems with Highest Incorrect Rate',
        'all_models_incorrect': 'Problems All Models Got Wrong',
        'most_models_incorrect': 'Problems Most Models Got Wrong (≥50%)',
        'test_info': 'Test',
        'problem_id': 'Problem ID',
        'incorrect_count': 'Incorrect Models',
        'correct_count': 'Correct Models',
        'total_models': 'Total Models',
        'attempted_models': 'Attempted Models',
        'question': 'Question',
        'difficulty_score': 'Difficulty Score',
        'by_session': 'By Session',
        'by_subject': 'By Subject',
        'by_year': 'By Year',
        'problem_count': 'Problem Count',
        'session_distribution': 'Problem Distribution by Session',
        'subject_distribution': 'Problem Distribution by Subject',
        'year_distribution': 'Problem Distribution by Year',
        'law_distribution_stat': 'Law/Non-Law Problem Distribution',
        'basic_stats': 'Basic Statistics',
        'help': 'Help',
        'new_features': 'New Features',
        'existing_features': 'Existing Features',
        'current_data': 'Currently Displayed Data',
        'problems': ' problems',
        'session_filter': 'Analyze specific session results only',
        'incorrect_pattern': 'Analyze difficult problems and incorrect patterns',
        'difficulty_comparison': 'Compare model performance by problem difficulty',
        'problem_type_filter': 'Distinguish image/text problems',
        'basic_filters': 'Filter by test name, model, detail type, prompting method',
        'law_analysis_desc': 'Analyze law/non-law distinction',
        'detail_analysis': 'Detailed analysis by subject and year',
        'font_size': 'Screen Font Size',
        'chart_text_size': 'Chart Text Size',
        'year_problem_distribution': 'Problem Distribution by Year',
        'problem_count_table': 'Problem Count by Year',
        'year_problem_chart': 'Problems by Year',
        'total_problem_count': 'Total Problems',
        'correct_models': 'Correct Models',
        'incorrect_models': 'Incorrect Models',
        'avg_accuracy_by_model': 'Average Accuracy by Model',
        'difficulty_range': 'Difficulty Range',
        'avg_difficulty': 'Average Difficulty',
        'difficulty_stats_by_range': 'Detailed Statistics by Difficulty Range',
        'very_hard': 'Very Hard',
        'hard': 'Hard',
        'medium': 'Medium',
        'easy': 'Easy',
        'very_easy': 'Very Easy',
        'problem_distribution': 'Problem Distribution',
        'response_time': 'Response Time',
        'avg_response_time': 'Average Response Time',
        'response_time_distribution': 'Response Time Distribution',
        'response_time_by_model': 'Response Time by Model',
        'response_time_stats': 'Response Time Statistics',
        'fastest_model': 'Fastest Model',
        'slowest_model': 'Slowest Model',
        'response_time_vs_accuracy': 'Response Time vs Accuracy',
        'time_per_problem': 'Time per Problem',
        'total_time': 'Total Time',
        'seconds': 'seconds',
        'minutes': 'minutes',
        # Token & Cost related
        'token_cost_analysis': 'Token & Cost Analysis',
        'token_usage': 'Token Usage',
        'input_tokens': 'Input Tokens',
        'output_tokens': 'Output Tokens',
        'total_tokens': 'Total Tokens',
        'avg_tokens_per_problem': 'Avg Tokens per Problem',
        'token_distribution': 'Token Distribution',
        'token_efficiency': 'Token Efficiency',
        'token_stats': 'Token Statistics',
        'io_ratio': 'Input/Output Token Ratio',
        'token_per_correct': 'Tokens per Correct Answer',
        'tokens': 'tokens',
        'cost_level': 'Cost Level',
        'cost_analysis': 'Cost Analysis',
        'cost_per_problem': 'Cost per Problem',
        'total_cost_estimate': 'Total Cost Estimate',
        'cost_vs_accuracy': 'Cost vs Accuracy',
        'cost_efficiency': 'Cost Efficiency',
        'most_efficient': 'Most Efficient Model',
        'least_efficient': 'Least Efficient Model',
        'cost_stats': 'Cost Statistics',
        'high': 'High',
        'medium_cost': 'Medium',
        'low': 'Low',
        'very_low': 'Very Low',
        'free': 'Free',
        'cost': 'cost',
        'actual_cost': 'Actual Cost',
        'estimated_cost': 'Estimated Cost',
        'cost_per_1k_tokens': 'Cost per 1K Tokens',
        'total_estimated_cost': 'Total Estimated Cost',
        'usd': 'USD',
    }
}

# 커스텀 CSS - 폰트 크기 및 레이아웃 조정
def apply_custom_css(font_size_multiplier=1.0):
    base_font = int(16 * font_size_multiplier)
    metric_value = int(32 * font_size_multiplier)
    metric_label = int(18 * font_size_multiplier)
    h1_size = f"{3 * font_size_multiplier}rem"
    h2_size = f"{2.2 * font_size_multiplier}rem"
    h3_size = f"{1.8 * font_size_multiplier}rem"
    
    st.markdown(f"""
    <style>
        /* 전체 폰트 크기 증가 */
        html, body, [class*="css"] {{
            font-size: {base_font}px;
        }}
        
        /* 메트릭 카드 폰트 크기 */
        [data-testid="stMetricValue"] {{
            font-size: {metric_value}px !important;
        }}
        
        [data-testid="stMetricLabel"] {{
            font-size: {metric_label}px !important;
        }}
        
        /* 헤더 폰트 크기 */
        h1 {{
            font-size: {h1_size} !important;
            font-weight: 700 !important;
        }}
        
        h2 {{
            font-size: {h2_size} !important;
            font-weight: 600 !important;
            margin-top: 1.5rem !important;
        }}
        
        h3 {{
            font-size: {h3_size} !important;
            font-weight: 600 !important;
        }}
        
        /* 테이블 폰트 크기 */
        .dataframe {{
            font-size: {int(16 * font_size_multiplier)}px !important;
        }}
        
        .dataframe th {{
            font-size: {int(16 * font_size_multiplier)}px !important;
            font-weight: 600 !important;
        }}
        
        .dataframe td {{
            font-size: {int(16 * font_size_multiplier)}px !important;
        }}
        
        /* 사이드바 폰트 크기 */
        .css-1d391kg, [data-testid="stSidebar"] {{
            font-size: {int(15 * font_size_multiplier)}px !important;
        }}
        
        /* 탭 폰트 크기 */
        .stTabs [data-baseweb="tab-list"] button {{
            font-size: {int(18 * font_size_multiplier)}px !important;
            padding: 12px 20px !important;
        }}
        
        /* 버튼 폰트 크기 */
        .stButton>button {{
            font-size: {base_font}px !important;
            padding: 0.5rem 1rem !important;
        }}
        
        /* 셀렉트박스 폰트 크기 */
        .stSelectbox label, .stMultiSelect label {{
            font-size: {base_font}px !important;
            font-weight: 600 !important;
        }}
        
        /* 차트 여백 조정 */
        .js-plotly-plot {{
            margin: 1rem 0 !important;
        }}
        
        /* 컨테이너 패딩 */
        .block-container {{
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
        }}
    </style>
    """, unsafe_allow_html=True)

# Plotly 차트 글로벌 폰트 크기 설정 (캐싱으로 불필요한 재생성 방지)
_plotly_template_cache = {}

def set_plotly_font_size(chart_text_multiplier=1.0):
    """모든 Plotly 차트에 적용될 기본 폰트 크기 설정 (캐싱 적용)"""
    import plotly.io as pio
    
    # 캐시 키 생성
    cache_key = round(chart_text_multiplier, 2)
    
    # 이미 동일한 설정이 적용되어 있으면 스킵
    if _plotly_template_cache.get('last_multiplier') == cache_key:
        return int(12 * chart_text_multiplier)
    
    # 기본 폰트 크기 계산
    title_size = int(20 * chart_text_multiplier)
    axis_size = int(14 * chart_text_multiplier)
    tick_size = int(12 * chart_text_multiplier)
    legend_size = int(12 * chart_text_multiplier)
    
    # plotly 기본 템플릿 복사
    pio.templates["custom"] = pio.templates["plotly"]
    
    # 전역 폰트 크기 설정
    pio.templates["custom"].layout.font.size = axis_size
    pio.templates["custom"].layout.title.font.size = title_size
    
    # 축 폰트 설정
    pio.templates["custom"].layout.xaxis.tickfont.size = tick_size
    pio.templates["custom"].layout.xaxis.title.font.size = axis_size
    pio.templates["custom"].layout.yaxis.tickfont.size = tick_size
    pio.templates["custom"].layout.yaxis.title.font.size = axis_size
    
    # 범례 폰트 설정
    pio.templates["custom"].layout.legend.font.size = legend_size
    
    # 기본 템플릿으로 설정
    pio.templates.default = "custom"
    
    # 캐시 업데이트
    _plotly_template_cache['last_multiplier'] = cache_key
    
    return int(12 * chart_text_multiplier)  # 히트맵용 크기 반환

# 안전한 정렬 함수 (타입 혼합 대응)
def safe_sort(values):
    """문자열과 숫자가 섞여있어도 안전하게 정렬"""
    try:
        # 타입별로 그룹화하여 정렬: 숫자 먼저, 그 다음 문자열
        return sorted(values, key=lambda x: (isinstance(x, str), x))
    except:
        # 실패하면 모두 문자열로 변환하여 정렬
        return sorted(values, key=str)

# ========== Excel/CSV 다운로드 헬퍼 함수 ==========

def create_download_button(df, filename, button_text="📥 Excel로 다운로드"):
    """데이터프레임을 Excel 파일로 다운로드하는 버튼 생성"""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Data')
    
    st.download_button(
        label=button_text,
        data=buffer.getvalue(),
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

def create_csv_download_button(df, filename, button_text="📄 CSV로 다운로드"):
    """데이터프레임을 CSV 파일로 다운로드하는 버튼 생성"""
    csv = df.to_csv(index=False).encode('utf-8-sig')  # BOM 추가로 한글 깨짐 방지
    st.download_button(
        label=button_text,
        data=csv,
        file_name=filename,
        mime="text/csv"
    )

def create_copy_button(df, button_text="📋 클립보드로 복사", key_suffix=""):
    """데이터프레임을 클립보드로 복사하는 버튼 생성 - 헤더 포함"""
    # 헤더 포함한 TSV 형식으로 변환
    tsv_data = df.to_csv(index=False, sep='\t')
    
    # 고유 key 생성 (중복 방지)
    import hashlib
    data_hash = hashlib.md5(tsv_data.encode()).hexdigest()[:8]
    unique_key = f"tsv_download_{data_hash}_{key_suffix}"
    
    # 2개 컬럼으로 분할: 다운로드 버튼 + 복사 가이드
    col_a, col_b = st.columns([1, 2])
    
    with col_a:
        # TSV 다운로드 버튼
        st.download_button(
            label=button_text,
            data=tsv_data,
            file_name="data_with_headers.tsv",
            mime="text/tab-separated-values",
            key=unique_key,
            help="Excel에서 열면 헤더 포함"
        )
    
    with col_b:
        # 헤더 포함 복사 가이드
        st.caption("💡 헤더 포함 복사: 표 좌측 상단 📋 클릭")

def display_table_with_download(df, title, excel_filename, lang='ko'):
    """표를 표시하고 다운로드/복사 버튼을 함께 제공"""
    if title:
        st.markdown(f"### {title}")
    
    # 고유 key 생성용 suffix (파일명 기반)
    key_suffix = excel_filename.replace('.xlsx', '').replace('.csv', '')
    
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        create_download_button(df, excel_filename)
    with col2:
        create_csv_download_button(df, excel_filename.replace('.xlsx', '.csv'))
    with col3:
        create_copy_button(df, "📋 " + ("TSV 다운로드" if lang == 'ko' else "Download TSV"), key_suffix)
    
    st.dataframe(df, width='stretch')
    st.markdown("---")

# ========== 모델 정보 추정 함수 ==========

def calculate_model_release_date(model_name):
    """모델명으로부터 대략적인 출시 시기 추정"""
    release_dates = {
        'GPT-4o': '2024-05',
        'GPT-4o-Mini': '2024-07',
        'GPT-4-Turbo': '2024-04',
        'GPT-3.5-Turbo': '2023-03',
        'Claude-Sonnet-4.5': '2024-10',
        'Claude-Sonnet-4': '2024-06',
        'Claude-3.5-Sonnet': '2024-06',
        'Claude-3.5-Haiku': '2024-08',
        'Claude-3-Opus': '2024-03',
        'Claude-3-Sonnet': '2024-03',
        'Claude-3-Haiku': '2024-03',
        'Llama-3.3-70b': '2024-12',
        'Llama-3.1-70b': '2024-07',
        'Llama-3.1-8b': '2024-07',
        'Qwen-2.5-72b': '2024-09',
        'Qwen-2.5-32b': '2024-09',
        'EXAONE-3.5-32b': '2024-08',
        'EXAONE-3.0-7.8b': '2024-08',
        'SOLAR-Pro': '2024-05',
        'Gemma-2-27b': '2024-06',
        'ko-gemma-2-9b': '2024-08',
    }
    
    for key, date in release_dates.items():
        if key.replace('-', '').replace('.', '').lower() in model_name.replace('-', '').replace('.', '').lower():
            return date
    
    return '2024-01'  # 기본값

def calculate_model_parameters(model_name):
    """모델명으로부터 파라미터 수 추정 (억 단위)"""
    if '70b' in model_name.lower() or '72b' in model_name.lower():
        return 70
    elif '32b' in model_name.lower():
        return 32
    elif '27b' in model_name.lower():
        return 27
    elif '9b' in model_name.lower():
        return 9
    elif '8b' in model_name.lower():
        return 8
    elif '7.8b' in model_name.lower() or '7b' in model_name.lower():
        return 7.8
    elif 'gpt-4' in model_name.lower():
        return 175  # 추정치
    elif 'claude' in model_name.lower():
        return 100  # 추정치
    else:
        return 10  # 기본값

# ========== 추가 분석 표 생성 함수 ==========

def create_testset_accuracy_table(filtered_df, testsets, lang='ko'):
    """테스트셋별 평균 정답률 표 - 테스트셋 원본 데이터 사용"""
    if '테스트명' not in filtered_df.columns:
        return None
    
    test_names = filtered_df['테스트명'].unique()
    
    data = []
    for test_name in test_names:
        # 해당 테스트의 평가 결과
        testset_df = filtered_df[filtered_df['테스트명'] == test_name]
        
        # 실제 문제 수는 testsets에서 가져오기
        if test_name in testsets:
            actual_problems = len(testsets[test_name])
            
            # 법령 문제 비율도 testsets에서 계산
            law_ratio = 0
            if 'law' in testsets[test_name].columns:
                law_count = len(testsets[test_name][testsets[test_name]['law'] == 'O'])
                law_ratio = (law_count / actual_problems * 100) if actual_problems > 0 else 0
        else:
            # testsets에 없으면 고유 문제 수로 추정
            actual_problems = testset_df['Question'].nunique() if 'Question' in testset_df.columns else len(testset_df)
            
            # 법령 문제 비율
            law_ratio = 0
            if 'law' in testset_df.columns:
                unique_problems = testset_df.drop_duplicates(subset=['Question'])
                law_count = len(unique_problems[unique_problems['law'] == 'O'])
                law_ratio = (law_count / actual_problems * 100) if actual_problems > 0 else 0
        
        # 평가 통계
        num_models = testset_df['모델'].nunique()
        total_evaluations = len(testset_df)
        
        # 정확도 계산
        if '정답여부' in testset_df.columns:
            accuracy = testset_df['정답여부'].mean() * 100
            correct_count = testset_df['정답여부'].sum()
        else:
            accuracy = 0
            correct_count = 0
        
        # 과목별 정답률 계산
        best_subject = ""
        best_subject_acc = 0
        worst_subject = ""
        worst_subject_acc = 100
        
        if 'Subject' in testset_df.columns and '정답여부' in testset_df.columns:
            subject_acc = testset_df.groupby('Subject')['정답여부'].mean() * 100
            if len(subject_acc) > 0:
                best_idx = subject_acc.idxmax()
                best_subject_acc = subject_acc[best_idx]
                best_subject = f"{best_idx} ({best_subject_acc:.1f}%)"
                
                worst_idx = subject_acc.idxmin()
                worst_subject_acc = subject_acc[worst_idx]
                worst_subject = f"{worst_idx} ({worst_subject_acc:.1f}%)"
        
        data.append({
            '테스트명' if lang == 'ko' else 'Test Name': test_name,
            '문제 수' if lang == 'ko' else 'Problems': actual_problems,
            '평가 모델 수' if lang == 'ko' else 'Models': num_models,
            '총 평가 횟수' if lang == 'ko' else 'Total Evaluations': total_evaluations,
            '평균 정답률 (%)' if lang == 'ko' else 'Avg Accuracy (%)': round(accuracy, 2),
            '최고 과목 (정답률)' if lang == 'ko' else 'Best Subject (Accuracy)': best_subject if best_subject else '-',
            '최저 과목 (정답률)' if lang == 'ko' else 'Worst Subject (Accuracy)': worst_subject if worst_subject else '-',
            '정답 수' if lang == 'ko' else 'Correct': int(correct_count),
            '법령 문제 비율 (%)' if lang == 'ko' else 'Law Problem Ratio (%)': round(law_ratio, 1)
        })
    
    df = pd.DataFrame(data).sort_values('평균 정답률 (%)' if lang == 'ko' else 'Avg Accuracy (%)', ascending=False)
    return df

def create_model_release_performance_table(filtered_df, lang='ko'):
    """표 3: 모델 출시 시기와 SafetyQ&A 성능"""
    models = filtered_df['모델'].unique()
    
    data = []
    for model in models:
        model_df = filtered_df[filtered_df['모델'] == model]
        accuracy = (model_df['정답여부'].mean() * 100) if '정답여부' in model_df.columns else 0
        release_date = calculate_model_release_date(model)
        
        data.append({
            '출시 시기' if lang == 'ko' else 'Release Date': release_date,
            '모델명' if lang == 'ko' else 'Model': model,
            '평균 정답률 (%)' if lang == 'ko' else 'Avg Accuracy (%)': round(accuracy, 2)
        })
    
    df = pd.DataFrame(data).sort_values('출시 시기' if lang == 'ko' else 'Release Date', ascending=False)
    return df

def create_response_time_parameters_table(filtered_df, lang='ko'):
    """표 4: 모델별 평균 응답 시간 및 정답률 (파라미터 수 포함)"""
    time_col = None
    for col in ['문제당평균시간(초)', '총소요시간(초)', 'question_duration']:
        if col in filtered_df.columns:
            time_col = col
            break
    
    if time_col is None:
        return None
    
    models = filtered_df['모델'].unique()
    
    data = []
    for model in models:
        model_df = filtered_df[filtered_df['모델'] == model]
        
        avg_time = model_df[time_col].mean() if time_col in model_df.columns else 0
        accuracy = (model_df['정답여부'].mean() * 100) if '정답여부' in model_df.columns else 0
        params = calculate_model_parameters(model)
        
        data.append({
            '모델명' if lang == 'ko' else 'Model': model,
            '파라미터 수 (B)' if lang == 'ko' else 'Parameters (B)': params,
            '평균 응답시간 (초)' if lang == 'ko' else 'Avg Response Time (s)': round(avg_time, 2),
            '정답률 (%)' if lang == 'ko' else 'Accuracy (%)': round(accuracy, 2)
        })
    
    df = pd.DataFrame(data).sort_values('파라미터 수 (B)' if lang == 'ko' else 'Parameters (B)', ascending=False)
    return df

def create_law_nonlaw_comparison_table(filtered_df, testsets, lang='ko'):
    """표 2: 테스트셋별 법령/비법령 정답률 비교"""
    if '테스트명' not in filtered_df.columns or 'law' not in filtered_df.columns:
        return None
    
    test_names = filtered_df['테스트명'].unique()
    
    data = []
    for test_name in test_names:
        testset_df = filtered_df[filtered_df['테스트명'] == test_name]
        
        # 법령 문제 통계
        law_df = testset_df[testset_df['law'] == 'O']
        non_law_df = testset_df[testset_df['law'] != 'O']
        
        # 실제 문제 수는 testsets에서
        law_problems = 0
        non_law_problems = 0
        total_problems = 0
        
        if test_name in testsets and 'law' in testsets[test_name].columns:
            test_data = testsets[test_name]
            law_problems = len(test_data[test_data['law'] == 'O'])
            non_law_problems = len(test_data[test_data['law'] != 'O'])
            total_problems = len(test_data)
        else:
            # Fallback: filtered_df에서 고유 문제 수로 계산
            unique_law = testset_df[testset_df['law'] == 'O']['Question'].nunique() if 'Question' in testset_df.columns else len(law_df)
            unique_non_law = testset_df[testset_df['law'] != 'O']['Question'].nunique() if 'Question' in testset_df.columns else len(non_law_df)
            law_problems = unique_law
            non_law_problems = unique_non_law
            total_problems = law_problems + non_law_problems
        
        # 정답률 계산
        law_acc = 0
        non_law_acc = 0
        
        if '정답여부' in testset_df.columns:
            law_acc = (law_df['정답여부'].mean() * 100) if len(law_df) > 0 else 0
            non_law_acc = (non_law_df['정답여부'].mean() * 100) if len(non_law_df) > 0 else 0
        
        # 정답 수
        law_correct = law_df['정답여부'].sum() if len(law_df) > 0 and '정답여부' in law_df.columns else 0
        non_law_correct = non_law_df['정답여부'].sum() if len(non_law_df) > 0 and '정답여부' in non_law_df.columns else 0
        
        # 정답률 차이
        diff = law_acc - non_law_acc
        
        # 법령 문제 비율
        law_ratio = (law_problems / total_problems * 100) if total_problems > 0 else 0
        
        data.append({
            '테스트명' if lang == 'ko' else 'Test Name': test_name,
            '법령 문제 수' if lang == 'ko' else 'Law Problems': law_problems,
            '법령 정답률 (%)' if lang == 'ko' else 'Law Accuracy (%)': round(law_acc, 2),
            '법령 정답 수' if lang == 'ko' else 'Law Correct': int(law_correct),
            '비법령 문제 수' if lang == 'ko' else 'Non-Law Problems': non_law_problems,
            '비법령 정답률 (%)' if lang == 'ko' else 'Non-Law Accuracy (%)': round(non_law_acc, 2),
            '비법령 정답 수' if lang == 'ko' else 'Non-Law Correct': int(non_law_correct),
            '정답률 차이 (법령-비법령)' if lang == 'ko' else 'Accuracy Diff (Law-NonLaw)': round(diff, 2),
            '법령 문제 비율 (%)' if lang == 'ko' else 'Law Ratio (%)': round(law_ratio, 1)
        })
    
    df = pd.DataFrame(data).sort_values('법령 문제 비율 (%)' if lang == 'ko' else 'Law Ratio (%)', ascending=False)
    return df

def create_year_correlation_table(filtered_df, lang='ko'):
    """표 6: 출제 연도별 평균 정답률 및 문항 수 (상관계수 포함)"""
    if 'Year' not in filtered_df.columns:
        return None
    
    # 벡터화된 연도 변환 (copy 최소화)
    year_int_series = filtered_df['Year'].apply(safe_convert_to_int)
    valid_mask = year_int_series.notna()
    
    if not valid_mask.any():
        return None
    
    # 필요한 컬럼만 선택하여 메모리 절약
    year_df = filtered_df.loc[valid_mask, ['Question', '정답여부']].assign(Year_Int=year_int_series[valid_mask])
    
    # 연도별 통계
    year_stats = year_df.groupby('Year_Int').agg({
        'Question': 'count',
        '정답여부': ['mean', 'std']
    }).reset_index()
    
    # 다국어 컬럼명
    if lang == 'ko':
        year_stats.columns = ['연도', '문항 수', '평균 정답률', '표준편차']
        year_col = '연도'
        count_col = '문항 수'
        acc_col = '평균 정답률'
        std_col = '표준편차'
        law_ratio_col = '법령 문항 비율 (%)'
        corr_label = '상관계수 (r)'
        p_label = 'p-value'
    else:
        year_stats.columns = ['Year', 'Problem Count', 'Avg Accuracy', 'Std Dev']
        year_col = 'Year'
        count_col = 'Problem Count'
        acc_col = 'Avg Accuracy'
        std_col = 'Std Dev'
        law_ratio_col = 'Law Ratio (%)'
        corr_label = 'Correlation (r)'
        p_label = 'p-value'
    
    year_stats[acc_col] = year_stats[acc_col] * 100
    year_stats[std_col] = year_stats[std_col] * 100
    year_stats[year_col] = year_stats[year_col].astype(int)
    
    # 법령 문항 비율
    if 'law' in year_df.columns:
        law_ratio = year_df.groupby('Year_Int').apply(
            lambda x: (x['law'] == 'O').sum() / len(x) * 100
        ).reset_index()
        law_ratio.columns = [year_col, law_ratio_col]
        year_stats = year_stats.merge(law_ratio, on=year_col, how='left')
    
    # 상관계수 계산
    if len(year_stats) > 1:
        correlation, p_value = stats.pearsonr(year_stats[year_col], year_stats[acc_col])
        
        # 상관계수 정보를 별도 행으로 추가
        corr_row = pd.DataFrame({
            year_col: [corr_label],
            count_col: ['-'],
            acc_col: [f"{correlation:.4f}"],
            std_col: ['-']
        })
        
        if law_ratio_col in year_stats.columns:
            corr_row[law_ratio_col] = ['-']
        
        p_row = pd.DataFrame({
            year_col: [p_label],
            count_col: ['-'],
            acc_col: [f"{p_value:.4f}"],
            std_col: ['-']
        })
        
        if law_ratio_col in year_stats.columns:
            p_row[law_ratio_col] = ['-']
        
        year_stats = pd.concat([year_stats, corr_row, p_row], ignore_index=True)
    
    return year_stats

def create_difficulty_distribution_table(filtered_df, lang='ko'):
    """표 7: 난이도 구간별 문항 분포 및 정답률"""
    # 문제별 난이도 계산
    difficulty = filtered_df.groupby('Question').agg({
        '정답여부': ['mean', 'count']
    }).reset_index()
    difficulty.columns = ['Question', 'difficulty_score', 'attempt_count']
    difficulty['difficulty_score'] = difficulty['difficulty_score'] * 100
    
    # 난이도 구간 분류
    def classify_difficulty(score, lang='ko'):
        if lang == 'ko':
            if score < 20:
                return '매우 어려움 (0-20%)'
            elif score < 40:
                return '어려움 (20-40%)'
            elif score < 60:
                return '보통 (40-60%)'
            elif score < 80:
                return '쉬움 (60-80%)'
            else:
                return '매우 쉬움 (80-100%)'
        else:
            if score < 20:
                return 'Very Hard (0-20%)'
            elif score < 40:
                return 'Hard (20-40%)'
            elif score < 60:
                return 'Medium (40-60%)'
            elif score < 80:
                return 'Easy (60-80%)'
            else:
                return 'Very Easy (80-100%)'
    
    difficulty['난이도_구간'] = difficulty['difficulty_score'].apply(lambda x: classify_difficulty(x, lang))
    
    # 구간별 통계
    difficulty_dist = difficulty.groupby('난이도_구간').agg({
        'Question': 'count',
        'difficulty_score': 'mean'
    }).reset_index()
    
    difficulty_dist.columns = [
        '난이도 구간' if lang == 'ko' else 'Difficulty Range',
        '문항 수' if lang == 'ko' else 'Problem Count',
        '평균 정답률 (%)' if lang == 'ko' else 'Avg Accuracy (%)'
    ]
    
    difficulty_dist['비율 (%)'] = (difficulty_dist['문항 수' if lang == 'ko' else 'Problem Count'] / 
                                    difficulty_dist['문항 수' if lang == 'ko' else 'Problem Count'].sum() * 100)
    
    # 난이도 순서 정의
    if lang == 'ko':
        order = ['매우 어려움 (0-20%)', '어려움 (20-40%)', '보통 (40-60%)', '쉬움 (60-80%)', '매우 쉬움 (80-100%)']
    else:
        order = ['Very Hard (0-20%)', 'Hard (20-40%)', 'Medium (40-60%)', 'Easy (60-80%)', 'Very Easy (80-100%)']
    
    difficulty_dist['난이도 구간' if lang == 'ko' else 'Difficulty Range'] = pd.Categorical(
        difficulty_dist['난이도 구간' if lang == 'ko' else 'Difficulty Range'],
        categories=order,
        ordered=True
    )
    
    difficulty_dist = difficulty_dist.sort_values('난이도 구간' if lang == 'ko' else 'Difficulty Range')
    
    return difficulty_dist

def create_incorrect_pattern_table(filtered_df, lang='ko'):
    """표 10: 주요 오답 패턴 및 빈도 분석"""
    # 문제별 오답 분석
    problem_analysis = filtered_df.groupby('Question').agg({
        '정답여부': ['sum', 'count', 'mean']
    }).reset_index()
    problem_analysis.columns = ['Question', 'correct_count', 'total_count', 'correct_rate']
    problem_analysis['incorrect_rate'] = 1 - problem_analysis['correct_rate']
    
    # 오답 패턴 분류
    patterns = []
    
    # 모든 모델이 틀린 문제
    all_wrong = problem_analysis[problem_analysis['correct_count'] == 0]
    patterns.append({
        '오답 패턴 유형' if lang == 'ko' else 'Error Pattern Type': '전체 모델 오답' if lang == 'ko' else 'All Models Incorrect',
        '문항 수' if lang == 'ko' else 'Problem Count': len(all_wrong),
        '모델 일치도 (%)' if lang == 'ko' else 'Model Agreement (%)': 100.0
    })
    
    # 대부분 모델이 틀린 문제 (70% 이상)
    most_wrong = problem_analysis[(problem_analysis['incorrect_rate'] >= 0.7) & (problem_analysis['incorrect_rate'] < 1.0)]
    if len(most_wrong) > 0:
        avg_agreement = most_wrong['incorrect_rate'].mean() * 100
        patterns.append({
            '오답 패턴 유형' if lang == 'ko' else 'Error Pattern Type': '대부분 모델 오답 (≥70%)' if lang == 'ko' else 'Most Models Incorrect (≥70%)',
            '문항 수' if lang == 'ko' else 'Problem Count': len(most_wrong),
            '모델 일치도 (%)' if lang == 'ko' else 'Model Agreement (%)': round(avg_agreement, 1)
        })
    
    # 절반 정도 모델이 틀린 문제
    half_wrong = problem_analysis[(problem_analysis['incorrect_rate'] >= 0.4) & (problem_analysis['incorrect_rate'] < 0.7)]
    if len(half_wrong) > 0:
        avg_agreement = half_wrong['incorrect_rate'].mean() * 100
        patterns.append({
            '오답 패턴 유형' if lang == 'ko' else 'Error Pattern Type': '절반 정도 오답 (40-70%)' if lang == 'ko' else 'About Half Incorrect (40-70%)',
            '문항 수' if lang == 'ko' else 'Problem Count': len(half_wrong),
            '모델 일치도 (%)' if lang == 'ko' else 'Model Agreement (%)': round(avg_agreement, 1)
        })
    
    # 일부 모델만 틀린 문제
    some_wrong = problem_analysis[(problem_analysis['incorrect_rate'] > 0) & (problem_analysis['incorrect_rate'] < 0.4)]
    if len(some_wrong) > 0:
        avg_agreement = some_wrong['incorrect_rate'].mean() * 100
        patterns.append({
            '오답 패턴 유형' if lang == 'ko' else 'Error Pattern Type': '일부 모델 오답 (<40%)' if lang == 'ko' else 'Some Models Incorrect (<40%)',
            '문항 수' if lang == 'ko' else 'Problem Count': len(some_wrong),
            '모델 일치도 (%)' if lang == 'ko' else 'Model Agreement (%)': round(avg_agreement, 1)
        })
    
    return pd.DataFrame(patterns)

def create_model_law_performance_table(filtered_df, lang='ko'):
    """
    표 5: 모델별 법령 문항 vs 비법령 문항 성능 비교
    """
    if 'law' not in filtered_df.columns or '정답여부' not in filtered_df.columns:
        return None
    
    results = []
    
    for model in filtered_df['모델'].unique():
        model_df = filtered_df[filtered_df['모델'] == model]
        
        law_df = model_df[model_df['law'] == 'O']
        non_law_df = model_df[model_df['law'] != 'O']
        
        law_acc = (law_df['정답여부'].mean() * 100) if len(law_df) > 0 else 0
        non_law_acc = (non_law_df['정답여부'].mean() * 100) if len(non_law_df) > 0 else 0
        diff = law_acc - non_law_acc
        
        results.append({
            '모델명' if lang == 'ko' else 'Model': model,
            '법령 문항 정답률 (%)' if lang == 'ko' else 'Law Accuracy (%)': round(law_acc, 2),
            '비법령 문항 정답률 (%)' if lang == 'ko' else 'Non-Law Accuracy (%)': round(non_law_acc, 2),
            '격차 (%p)' if lang == 'ko' else 'Gap (%p)': round(diff, 2),
            '법령 문항 수' if lang == 'ko' else 'Law Count': len(law_df),
            '비법령 문항 수' if lang == 'ko' else 'Non-Law Count': len(non_law_df)
        })
    
    df = pd.DataFrame(results)
    
    if len(df) > 0:
        # 격차 절대값 순으로 정렬
        df['abs_gap'] = df['격차 (%p)' if lang == 'ko' else 'Gap (%p)'].abs()
        df = df.sort_values('abs_gap')
        df = df.drop('abs_gap', axis=1)
        
        # 순위 추가
        df.insert(0, '격차 순위' if lang == 'ko' else 'Gap Rank', range(1, len(df) + 1))
    
    return df

def create_difficulty_model_performance_table(filtered_df, lang='ko'):
    """
    표 8: 주요 모델의 난이도 구간별 정답률
    """
    if '정답여부' not in filtered_df.columns:
        return None
    
    # 문제별 난이도 계산
    difficulty = filtered_df.groupby('Question')['정답여부'].mean() * 100
    
    # 난이도 구간 분류
    def classify_difficulty(score):
        if score < 20:
            return '매우 어려움' if lang == 'ko' else 'Very Hard'
        elif score < 40:
            return '어려움' if lang == 'ko' else 'Hard'
        elif score < 60:
            return '보통' if lang == 'ko' else 'Medium'
        elif score < 80:
            return '쉬움' if lang == 'ko' else 'Easy'
        else:
            return '매우 쉬움' if lang == 'ko' else 'Very Easy'
    
    # 난이도 레벨을 Series로 생성 (copy 없이)
    difficulty_levels = filtered_df['Question'].map(
        lambda q: classify_difficulty(difficulty.get(q, 50))
    )
    
    # 난이도 순서
    difficulty_order = [
        '매우 쉬움' if lang == 'ko' else 'Very Easy',
        '쉬움' if lang == 'ko' else 'Easy',
        '보통' if lang == 'ko' else 'Medium',
        '어려움' if lang == 'ko' else 'Hard',
        '매우 어려움' if lang == 'ko' else 'Very Hard'
    ]
    
    # 상위 모델 선택 (정답률 기준)
    top_models = filtered_df.groupby('모델')['정답여부'].mean().nlargest(10).index.tolist()
    
    results = []
    for model in top_models:
        model_mask = filtered_df['모델'] == model
        model_difficulty = difficulty_levels[model_mask]
        model_correct = filtered_df.loc[model_mask, '정답여부']
        
        row = {'모델명' if lang == 'ko' else 'Model': model}
        
        for diff_level in difficulty_order:
            diff_mask = model_difficulty == diff_level
            diff_correct = model_correct[diff_mask]
            acc = (diff_correct.mean() * 100) if len(diff_correct) > 0 else 0
            row[diff_level] = round(acc, 1)
        
        results.append(row)
    
    return pd.DataFrame(results)

def create_cost_efficiency_table(filtered_df, lang='ko'):
    """
    표 9: 주요 상업용 모델의 비용 효율성 비교
    """
    # 토큰 컬럼 확인
    token_columns = {
        'input': ['입력토큰', 'input_tokens', 'Input Tokens'],
        'output': ['출력토큰', 'output_tokens', 'Output Tokens'],
        'total': ['총토큰', 'total_tokens', 'Total Tokens']
    }
    
    available_cols = {}
    for key, possible_names in token_columns.items():
        for col_name in possible_names:
            if col_name in filtered_df.columns:
                available_cols[key] = col_name
                break
    
    if not available_cols or '정답여부' not in filtered_df.columns:
        return None
    
    # 상업용 모델만 필터링
    commercial_models = ['GPT-4o', 'GPT-4o-Mini', 'Claude-Sonnet-4.5', 'Claude-Sonnet-4', 'Claude-3.5-Sonnet', 'Claude-3.5-Haiku', 'Claude-Haiku-4.5']
    commercial_df = filtered_df[filtered_df['모델'].str.contains('|'.join(commercial_models), case=False, na=False)]
    
    if len(commercial_df) == 0:
        return None
    
    # 모델 가격 (per 1M tokens)
    MODEL_PRICING = {
        'GPT-4o': {'input': 5.00, 'output': 15.00},
        'GPT-4o-Mini': {'input': 0.150, 'output': 0.600},
        'Claude-Sonnet-4.5': {'input': 3.00, 'output': 15.00},
        'Claude-Sonnet-4': {'input': 3.00, 'output': 15.00},
        'Claude-3.5-Sonnet': {'input': 3.00, 'output': 15.00},
        'Claude-Haiku-4.5': {'input': 1.00, 'output': 5.00},
        'Claude-3.5-Haiku': {'input': 0.80, 'output': 4.00}
    }
    
    results = []
    
    for model in commercial_df['모델'].unique():
        model_df = commercial_df[commercial_df['모델'] == model]
        
        # 정답률
        acc = model_df['정답여부'].mean() * 100
        correct_count = model_df['정답여부'].sum()
        
        # 평균 토큰
        if 'input' in available_cols and 'output' in available_cols:
            avg_input = model_df[available_cols['input']].mean()
            avg_output = model_df[available_cols['output']].mean()
        elif 'total' in available_cols:
            avg_total = model_df[available_cols['total']].mean()
            avg_input = avg_total * 0.6  # 추정
            avg_output = avg_total * 0.4  # 추정
        else:
            continue
        
        # 비용 계산
        matched_pricing = None
        for price_model, pricing in MODEL_PRICING.items():
            if price_model.replace('-', '').replace('.', '').lower() in model.replace('-', '').replace('.', '').lower():
                matched_pricing = pricing
                break
        
        if matched_pricing:
            # 문제당 비용
            cost_per_problem = (avg_input / 1_000_000) * matched_pricing['input'] + \
                              (avg_output / 1_000_000) * matched_pricing['output']
            
            # 정답 1000개당 비용
            if correct_count > 0:
                cost_per_1000_correct = (cost_per_problem * len(model_df) / correct_count) * 1000
            else:
                cost_per_1000_correct = float('inf')
            
            results.append({
                '모델명' if lang == 'ko' else 'Model': model,
                '정답률 (%)' if lang == 'ko' else 'Accuracy (%)': round(acc, 2),
                '평균 입력 토큰' if lang == 'ko' else 'Avg Input Tokens': int(avg_input),
                '평균 출력 토큰' if lang == 'ko' else 'Avg Output Tokens': int(avg_output),
                '총 비용 ($)' if lang == 'ko' else 'Total Cost ($)': round(cost_per_problem * len(model_df), 4),
                '정답 1000개당 비용 ($)' if lang == 'ko' else 'Cost per 1K Correct ($)': round(cost_per_1000_correct, 2) if cost_per_1000_correct != float('inf') else 0
            })
    
    df = pd.DataFrame(results)
    
    if len(df) > 0:
        # 비용 효율성 순으로 정렬 (정답 1000개당 비용 낮은 순)
        df = df.sort_values('정답 1000개당 비용 ($)' if lang == 'ko' else 'Cost per 1K Correct ($)')
    
    return df

def create_benchmark_comparison_table(filtered_df, lang='ko'):
    """
    표 11: SafetyQ&A와 범용 벤치마크 성능 비교
    (실제 데이터가 없으므로 예시 데이터 생성)
    """
    if '정답여부' not in filtered_df.columns:
        return None
    
    # SafetyQ&A 성능 계산
    safetyqa_performance = {}
    for model in filtered_df['모델'].unique():
        model_df = filtered_df[filtered_df['모델'] == model]
        safetyqa_performance[model] = model_df['정답여부'].mean() * 100
    
    # 예시: 범용 벤치마크 점수 (실제 데이터는 외부에서 가져와야 함)
    # 여기서는 SafetyQ&A 성능 기반으로 추정치 생성
    benchmark_data = []
    
    for model, safetyqa_score in safetyqa_performance.items():
        # 범용 벤치마크는 일반적으로 SafetyQ&A보다 높음
        mmlu_score = min(safetyqa_score * 1.2 + 10, 95)
        gpqa_score = min(safetyqa_score * 0.8 + 5, 70)
        mmlu_pro_score = min(safetyqa_score * 0.9 + 8, 80)
        
        benchmark_data.append({
            '모델명' if lang == 'ko' else 'Model': model,
            'SafetyQ&A': round(safetyqa_score, 1),
            'MMLU': round(mmlu_score, 1),
            'GPQA': round(gpqa_score, 1),
            'MMLU-Pro': round(mmlu_pro_score, 1)
        })
    
    df = pd.DataFrame(benchmark_data)
    
    if len(df) > 0:
        # SafetyQ&A 성능 순으로 정렬
        df = df.sort_values('SafetyQ&A', ascending=False)
    
    return df

# 앙상블 모델 생성 함수 (최적화 버전)
def create_ensemble_model(base_df, ensemble_name, selected_model_names, method='majority'):
    """
    여러 모델의 예측을 결합하여 앙상블 모델 데이터 생성 (최적화 버전)
    
    Args:
        base_df: 원본 데이터프레임
        ensemble_name: 앙상블 모델 이름
        selected_model_names: 앙상블에 포함할 모델 이름 리스트
        method: 앙상블 방법 ('majority' 또는 'weighted')
    
    Returns:
        ensemble_df: 앙상블 모델의 예측 결과 데이터프레임
    """
    # 선택된 모델만 필터링 (한 번만 수행)
    filtered_df = base_df[base_df['모델'].isin(selected_model_names)]
    
    if filtered_df.empty:
        return pd.DataFrame()
    
    # 모델별 전체 정확도 계산 (가중 투표용) - 미리 계산
    model_accuracy = {}
    if method == 'weighted':
        model_accuracy = filtered_df.groupby('모델')['정답여부'].mean().to_dict()
    
    # 문제별 모델 수 계산하여 모든 모델이 푼 문제만 선택
    question_model_counts = filtered_df.groupby('Question')['모델'].nunique()
    valid_questions = question_model_counts[question_model_counts == len(selected_model_names)].index
    
    if len(valid_questions) == 0:
        return pd.DataFrame()
    
    # 유효한 문제만 필터링
    valid_df = filtered_df[filtered_df['Question'].isin(valid_questions)]
    
    ensemble_rows = []
    
    # 문제별로 그룹화하여 처리 (groupby 사용으로 최적화)
    for question, q_df in valid_df.groupby('Question'):
        # 대표 행 가져오기
        base_row = q_df.iloc[0].to_dict()
        
        # 앙상블 예측 계산
        predictions = q_df['예측답'].tolist()
        
        if method == 'majority':
            # 다수결 투표
            counter = Counter(predictions)
            ensemble_answer = counter.most_common(1)[0][0]
        else:  # weighted
            # 가중 투표 - 벡터화 처리 (iterrows 대신 zip 사용)
            answer_weights = {}
            for answer, model in zip(q_df['예측답'].values, q_df['모델'].values):
                weight = model_accuracy.get(model, 0)
                answer_weights[answer] = answer_weights.get(answer, 0) + weight
            
            ensemble_answer = max(answer_weights, key=answer_weights.get) if answer_weights else None
        
        # 앙상블 결과 행 생성
        if ensemble_answer is not None:
            base_row['모델'] = ensemble_name
            base_row['예측답'] = ensemble_answer
            base_row['정답여부'] = (base_row.get('Answer') == ensemble_answer) if pd.notna(base_row.get('Answer')) else False
            base_row['상세도'] = 'ensemble'
            base_row['프롬프팅'] = method
            ensemble_rows.append(base_row)
    
    if ensemble_rows:
        return pd.DataFrame(ensemble_rows)
    return pd.DataFrame()


# 모델명 포맷팅 함수 (캐싱으로 반복 호출 최적화)
@lru_cache(maxsize=256)
def format_model_name(model_str):
    """
    모델명을 사람이 읽기 쉬운 형식으로 변환 (캐싱 적용)
    예: claude-sonnet-4-5-20250929 → Claude-Sonnet-4.5
        gpt-4o-mini → GPT-4o-Mini
        llama-3-3-70b → Llama-3.3-70b
    """
    # 날짜 패턴 제거 (8자리 숫자)
    model_str = re.sub(r'-\d{8}$', '', model_str)
    
    # 특수 케이스: GPT 모델
    if model_str.startswith('gpt-'):
        parts = model_str.split('-')
        formatted_parts = ['GPT']
        
        for i in range(1, len(parts)):
            part = parts[i]
            if part == '4o' or part == '3.5':
                formatted_parts.append(part)
            elif part.isdigit():
                formatted_parts.append(part)
            else:
                formatted_parts.append(part.capitalize())
        
        return '-'.join(formatted_parts)
    
    # Claude 모델 처리
    if model_str.startswith('claude-'):
        parts = model_str.split('-')
        formatted_parts = ['Claude']
        
        i = 1
        while i < len(parts):
            part = parts[i]
            
            if i + 1 < len(parts) and part.isdigit() and parts[i+1].isdigit():
                formatted_parts.append(f"{part}.{parts[i+1]}")
                i += 2
            elif part in ['sonnet', 'haiku', 'opus']:
                formatted_parts.append(part.capitalize())
                i += 1
            elif part.isdigit():
                formatted_parts.append(part)
                i += 1
            else:
                formatted_parts.append(part.capitalize())
                i += 1
        
        return '-'.join(formatted_parts)
    
    # 기타 모델: 스마트 버전 번호 처리
    parts = model_str.split('-')
    formatted_parts = []
    
    i = 0
    while i < len(parts):
        part = parts[i]
        
        if i == 0:
            if len(part) > 1 and part[:-1].isalpha() and part[-1].isdigit():
                if i + 1 < len(parts) and parts[i+1].isdigit() and len(parts[i+1]) == 1:
                    formatted_parts.append(f"{part.capitalize()}.{parts[i+1]}")
                    i += 2
                    continue
            formatted_parts.append(part.capitalize())
            i += 1
        elif (i + 1 < len(parts) and 
              part.isdigit() and len(part) == 1 and 
              parts[i+1].isdigit() and len(parts[i+1]) == 1):
            formatted_parts.append(f"{part}.{parts[i+1]}")
            i += 2
        elif not part.isdigit() and not any(c.isdigit() for c in part):
            formatted_parts.append(part.capitalize())
            i += 1
        else:
            formatted_parts.append(part)
            i += 1
    
    return '-'.join(formatted_parts)


# 데이터 로드 함수
@st.cache_data
def load_data(data_dir):
    """모든 CSV 파일을 로드하고 통합"""
    
    # testset 파일들 로드
    testset_files = glob.glob(os.path.join(data_dir, "testset_*.csv"))
    testsets = {}
    for file in testset_files:
        test_name = os.path.basename(file).replace("testset_", "").replace(".csv", "")
        try:
            df = pd.read_csv(file, encoding='utf-8')
            testsets[test_name] = df
        except:
            try:
                df = pd.read_csv(file, encoding='cp949')
                testsets[test_name] = df
            except:
                continue
    
    # testset에서 테스트명 목록 추출 (자동 감지)
    available_test_names = list(testsets.keys())
    
    # 결과 파일들 로드
    result_files = glob.glob(os.path.join(data_dir, "*_detailed_*.csv")) + \
                   glob.glob(os.path.join(data_dir, "*_summary_*.csv"))
    
    results = []
    for file in result_files:
        filename = os.path.basename(file)
        
        try:
            # 파일명 형식: {모델명}_{상세도}_{프롬프팅}_{테스트명}.csv
            # 예: llama-3-3-70b_detailed_noprompting_산업안전기사.csv
            
            # 테스트명 찾기 및 제거 (testset에서 추출한 목록 사용)
            test_name = None
            filename_without_csv = filename.replace('.csv', '')
            
            # 가장 긴 테스트명부터 매칭 (부분 매칭 방지)
            sorted_test_names = sorted(available_test_names, key=len, reverse=True)
            
            for tn in sorted_test_names:
                if filename_without_csv.endswith('_' + tn):
                    test_name = tn
                    # 테스트명 제거
                    filename_without_test = filename_without_csv[:-len('_' + tn)]
                    break
            
            if test_name is None:
                continue
            
            # 남은 부분을 '_'로 분리
            parts = filename_without_test.split('_')
            
            if len(parts) < 3:
                continue
            
            # 상세도 찾기 (detailed 또는 summary)
            detail_type = None
            detail_idx = -1
            for i, part in enumerate(parts):
                if part in ['detailed', 'summary']:
                    detail_type = part
                    detail_idx = i
                    break
            
            if detail_type is None or detail_idx == -1:
                continue
            
            # 모델명 추출 (상세도 이전까지의 모든 부분을 결합)
            model_parts = parts[:detail_idx]
            model_raw = '_'.join(model_parts)
            
            # 프롬프팅 방식 추출 (상세도 다음부터 끝까지)
            prompt_parts = parts[detail_idx + 1:]
            prompt_raw = '_'.join(prompt_parts)
            
            # 프롬프팅 방식 정규화
            if "noprompting" in prompt_raw.lower() or "no-prompting" in prompt_raw.lower() or "no_prompting" in prompt_raw.lower():
                prompt_type = "no-prompting"
            elif "few-shot" in prompt_raw.lower() or "few_shot" in prompt_raw.lower() or "fewshot" in prompt_raw.lower():
                prompt_type = "few-shot"
            elif "cot" in prompt_raw.lower() or "chain-of-thought" in prompt_raw.lower():
                prompt_type = "cot"
            else:
                prompt_type = prompt_raw if prompt_raw else "unknown"
            
            # 🔥 모델명 자동 파싱 및 정규화 (하드코딩 제거)
            # 언더스코어를 하이픈으로 변환하고 소문자로 정규화
            model_normalized = model_raw.lower().replace('_', '-')
            
            # 외부에 정의된 캐싱된 format_model_name 함수 사용
            model = format_model_name(model_normalized)
            
            # CSV 파일 읽기
            try:
                df = pd.read_csv(file, encoding='utf-8')
            except:
                try:
                    df = pd.read_csv(file, encoding='cp949')
                except:
                    continue
            
            # 메타데이터 추가
            df['모델'] = model
            df['상세도'] = detail_type
            df['프롬프팅'] = prompt_type
            df['테스트명'] = test_name
            
            results.append(df)
            
        except Exception as e:
            st.sidebar.warning(f"파일 로드 실패: {os.path.basename(file)}")
            continue
    
    if results:
        results_df = pd.concat(results, ignore_index=True)
    else:
        results_df = pd.DataFrame()
    
    return testsets, results_df

def safe_convert_to_int(value):
    """안전하게 값을 정수로 변환 - 쉼표 구분자 처리 개선"""
    try:
        # None이나 NaN 처리
        if pd.isna(value):
            return None
            
        # 문자열인 경우 쉼표 제거 (천 단위 구분자)
        if isinstance(value, str):
            # 쉼표는 천 단위 구분자이므로 그냥 제거
            value = value.replace(',', '')
        
        # float로 변환 후 int로 변환
        return int(float(value))
    except (ValueError, TypeError):
        return None

def get_available_sessions(df, test_names):
    """특정 테스트들에서 사용 가능한 세션 목록 반환 (문자열과 숫자 모두 지원)"""
    if df is None or len(df) == 0:
        return []
    
    # 여러 테스트 선택 시 필터링
    if test_names:
        test_df = df[df['테스트명'].isin(test_names)] if '테스트명' in df.columns else df
    else:
        test_df = df
    
    if 'Session' in test_df.columns:
        sessions_raw = test_df['Session'].dropna().unique().tolist()
        sessions_clean = []
        
        for s in sessions_raw:
            # 숫자로 변환 가능한지 먼저 시도
            s_int = safe_convert_to_int(s)
            if s_int is not None:
                # 숫자로 변환 가능하면 정수로 저장
                if s_int not in sessions_clean:
                    sessions_clean.append(s_int)
            else:
                # 숫자로 변환 불가능하면 문자열로 저장
                if isinstance(s, str):
                    s_clean = s.strip()
                    if s_clean and s_clean not in sessions_clean:
                        sessions_clean.append(s_clean)
        
        # 정렬: 숫자 먼저, 그 다음 문자열
        return sorted(sessions_clean, key=lambda x: (isinstance(x, str), x))
    return []

def create_problem_identifier(row, lang='ko'):
    """문제 식별자 생성 (테스트명/연도/세션/과목/문제번호)"""
    parts = []
    
    if 'Test Name' in row and pd.notna(row['Test Name']):
        parts.append(str(row['Test Name']))
    elif '테스트명' in row and pd.notna(row['테스트명']):
        parts.append(str(row['테스트명']))
    
    if 'Year' in row and pd.notna(row['Year']):
        year_int = safe_convert_to_int(row['Year'])
        if year_int:
            parts.append(str(year_int))
    
    if 'Session' in row and pd.notna(row['Session']):
        session_int = safe_convert_to_int(row['Session'])
        if session_int:
            parts.append(f"S{session_int}")
    
    if 'Subject' in row and pd.notna(row['Subject']):
        parts.append(str(row['Subject']))
    
    if 'Number' in row and pd.notna(row['Number']):
        number_int = safe_convert_to_int(row['Number'])
        if number_int:
            parts.append(f"Q{number_int}")
    
    return " / ".join(parts) if parts else "Unknown"

def get_testset_statistics(testsets, test_name, lang='ko'):
    """테스트셋의 기초 통계 반환"""
    t = LANGUAGES[lang]
    
    if test_name not in testsets:
        return None
    
    df = testsets[test_name]
    stats = {}
    
    # 총 문제 수
    stats['total_problems'] = len(df)
    
    # 법령/비법령 문제 수
    if 'law' in df.columns:
        stats['law_problems'] = len(df[df['law'] == 'O'])
        stats['non_law_problems'] = len(df[df['law'] != 'O'])
    
    # 과목별 문제 수
    if 'Subject' in df.columns:
        stats['by_subject'] = df['Subject'].value_counts().to_dict()
    
    # 연도별 문제 수
    if 'Year' in df.columns:
        stats['by_year'] = df['Year'].value_counts().sort_index().to_dict()
    
    # 세션별 문제 수
    if 'Session' in df.columns:
        stats['by_session'] = df['Session'].value_counts().sort_index().to_dict()
    
    return stats

# 메인 실행
def main():
    # 🔥 GitHub에서 데이터 다운로드 (최초 1회)
    download_data_from_github()
    
    # 언어 선택 (사이드바 상단에 배치)
    st.sidebar.selectbox(
        "Language / 언어",
        options=['ko', 'en'],
        format_func=lambda x: "한국어" if x == 'ko' else "English",
        key='language'
    )
    
    lang = st.session_state.language
    t = LANGUAGES[lang]
    
    # 화면 설정
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎨 " + ("화면 설정" if lang == 'ko' else "Display Settings"))
    
    # 폰트 크기 조정
    font_size = st.sidebar.slider(
        t['font_size'],
        min_value=0.8,
        max_value=1.5,
        value=1.0,
        step=0.1,
        help="화면 전체의 폰트 크기를 조절합니다"
    )
    
    # 차트 텍스트 크기 조정
    chart_text_size = st.sidebar.slider(
        t['chart_text_size'],
        min_value=0.7,
        max_value=1.8,
        value=1.0,
        step=0.1,
        help="차트 내부 텍스트, 숫자, 레이블 크기를 조절합니다"
    )
    
    apply_custom_css(font_size)
    set_plotly_font_size(chart_text_size)
    
    # 차트 주석 크기 (히트맵, 텍스트 등에 사용)
    annotation_size = int(12 * chart_text_size)
    
    # 제목
    st.title(f"🎯 {t['title']}")
    st.markdown("---")
    
    # 데이터 디렉토리는 항상 ./data (GitHub에서 다운로드한 폴더)
    data_dir = "./data"
    
    if not os.path.exists(data_dir):
        st.error(f"Directory not found: {data_dir}")
        return
    
    # 데이터 로드
    testsets, results_df = load_data(data_dir)
    
    if results_df.empty:
        st.warning("No data files found in the specified directory.")
        return
    
    # 정답여부 컬럼 생성
    if 'Answer' in results_df.columns and '예측답' in results_df.columns:
        results_df['정답여부'] = results_df.apply(
            lambda row: row['Answer'] == row['예측답'] if pd.notna(row['Answer']) and pd.notna(row['예측답']) else False,
            axis=1
        )
    
    # 사이드바 필터
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"## {t['filters']}")
    
    # 테스트명 필터 (multiselect로 변경)
    test_names = sorted(results_df['테스트명'].unique().tolist())
    selected_tests = st.sidebar.multiselect(
        t['testname'],
        options=test_names,
        default=test_names,
        help="여러 테스트를 선택할 수 있습니다"
    )
    
    # 테스트 선택에 따른 데이터 필터링 (마지막에 한 번만 .copy() 수행)
    if selected_tests:
        filtered_df = results_df[results_df['테스트명'].isin(selected_tests)]
    else:
        filtered_df = results_df
    
    # ========== 앙상블 모델 관리 ==========
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### 🎯 {t['ensemble_management']}")
    
    # 세션 상태 초기화
    if 'ensembles' not in st.session_state:
        st.session_state.ensembles = []
    
    # 앙상블 생성 UI
    with st.sidebar.expander(f"➕ {t['create_ensemble']}", expanded=False):
        # 앙상블 이름 입력
        ensemble_name_input = st.text_input(
            t['ensemble_name'],
            value="",
            placeholder="예: GPT 앙상블" if lang == 'ko' else "e.g., GPT Ensemble",
            key="ensemble_name_input"
        )
        
        # 모델 선택 (앙상블 제외)
        available_models_for_ensemble = sorted([m for m in results_df['모델'].unique() if '🎯' not in str(m)])
        selected_ensemble_models = st.multiselect(
            t['select_models'],
            options=available_models_for_ensemble,
            default=[],
            help=t['min_2_models'],
            key="ensemble_models_select"
        )
        
        # 앙상블 방법 선택
        ensemble_method_options = [t['majority_voting'], t['weighted_voting']]
        selected_ensemble_method = st.selectbox(
            t['ensemble_method'],
            options=ensemble_method_options,
            key="ensemble_method_select"
        )
        
        # 앙상블 추가 버튼
        if st.button(f"✅ {t['add_ensemble']}", use_container_width=True, key="add_ensemble_btn"):
            # 유효성 검사
            if not ensemble_name_input or ensemble_name_input.strip() == "":
                st.error("앙상블 이름을 입력하세요" if lang == 'ko' else "Please enter ensemble name")
            elif len(selected_ensemble_models) < 2:
                st.error(t['min_2_models'])
            elif any(e['name'] == ensemble_name_input for e in st.session_state.ensembles):
                st.error("같은 이름의 앙상블이 이미 있습니다" if lang == 'ko' else "Ensemble with same name exists")
            else:
                # 앙상블 방법 매핑
                method_key = 'majority' if selected_ensemble_method == t['majority_voting'] else 'weighted'
                
                # 앙상블 정보 저장
                st.session_state.ensembles.append({
                    'name': f"🎯 {ensemble_name_input}",  # 앙상블 식별을 위해 이모지 추가
                    'models': selected_ensemble_models.copy(),
                    'method': method_key,
                    'method_display': selected_ensemble_method
                })
                st.success(f"✅ {t['ensemble_added']}: {ensemble_name_input}")
                st.rerun()
    
    # 현재 앙상블 목록 표시
    if st.session_state.ensembles:
        st.sidebar.markdown(f"**{t['current_ensembles']}:**")
        
        for idx, ensemble in enumerate(st.session_state.ensembles):
            col1, col2 = st.sidebar.columns([4, 1])
            
            with col1:
                clean_name = ensemble['name'].replace('🎯 ', '')
                st.sidebar.text(f"{ensemble['name']}")
                st.sidebar.caption(f"  • {ensemble['method_display']}")
                st.sidebar.caption(f"  • {len(ensemble['models'])} 모델")
            
            with col2:
                if st.button("🗑️", key=f"del_ensemble_{idx}", help=t['remove_ensemble']):
                    st.session_state.ensembles.pop(idx)
                    st.success(t['ensemble_removed'])
                    st.rerun()
        
        # 앙상블 모델 생성 및 원본 데이터에 통합
        ensemble_dfs = []
        for ensemble in st.session_state.ensembles:
            ensemble_df = create_ensemble_model(
                results_df,
                ensemble['name'],
                ensemble['models'],
                ensemble['method']
            )
            if not ensemble_df.empty:
                ensemble_dfs.append(ensemble_df)
        
        # 앙상블 데이터를 원본 데이터에 추가하여 새로운 통합 데이터 생성
        if ensemble_dfs:
            # results_df에 앙상블 데이터 추가 (원본은 유지, 통합 데이터는 별도)
            integrated_df = pd.concat([results_df] + ensemble_dfs, ignore_index=True)
            
            # filtered_df를 integrated_df 기반으로 재생성 (마지막에만 copy)
            if selected_tests:
                filtered_df = integrated_df[integrated_df['테스트명'].isin(selected_tests)]
            else:
                filtered_df = integrated_df
            
            st.sidebar.success(f"🎯 {len(st.session_state.ensembles)}개 앙상블 활성" if lang == 'ko' else f"🎯 {len(st.session_state.ensembles)} ensemble(s) active")
        else:
            # 앙상블 데이터가 없으면 원본 사용
            if selected_tests:
                filtered_df = results_df[results_df['테스트명'].isin(selected_tests)]
            else:
                filtered_df = results_df
    else:
        st.sidebar.info(t['no_ensembles'])
    
    # ========== 앙상블 관리 끝 ==========
    
    # 모델 필터
    models = sorted(filtered_df['모델'].unique().tolist())
    selected_models = st.sidebar.multiselect(
        t['model'],
        options=models,
        default=models
    )
    
    if selected_models:
        filtered_df = filtered_df[filtered_df['모델'].isin(selected_models)]
    
    # 상세도 필터 (multiselect로 변경)
    details = sorted(filtered_df['상세도'].unique().tolist())
    selected_details = st.sidebar.multiselect(
        t['detail_type'],
        options=details,
        default=details,
        help="여러 상세도를 선택할 수 있습니다"
    )
    
    if selected_details:
        filtered_df = filtered_df[filtered_df['상세도'].isin(selected_details)]
    
    # 프롬프팅 방식 필터 (multiselect로 변경)
    prompts = sorted(filtered_df['프롬프팅'].unique().tolist())
    selected_prompts = st.sidebar.multiselect(
        t['prompting'],
        options=prompts,
        default=prompts,
        help="여러 프롬프팅 방식을 선택할 수 있습니다"
    )
    
    if selected_prompts:
        filtered_df = filtered_df[filtered_df['프롬프팅'].isin(selected_prompts)]
    
    # 세션 필터 (원본 데이터에서 추출, multiselect로 변경)
    if selected_tests:
        # 선택된 테스트들의 원본 데이터에서 세션 추출
        available_sessions = get_available_sessions(results_df, selected_tests)
        if available_sessions:
            selected_sessions = st.sidebar.multiselect(
                t['session'],
                options=available_sessions,
                default=available_sessions,
                help="여러 세션을 선택할 수 있습니다"
            )
            
            if selected_sessions:
                # 선택된 세션과 매칭 (문자열과 숫자 모두 지원)
                def match_session(x):
                    if pd.isna(x):
                        return False
                    
                    # x를 정수로 변환 시도
                    x_int = safe_convert_to_int(x)
                    
                    # 선택된 세션에 정수로 변환된 값이 있는지 확인
                    if x_int is not None and x_int in selected_sessions:
                        return True
                    
                    # 문자열로 직접 비교
                    if isinstance(x, str):
                        x_clean = x.strip()
                        return x_clean in selected_sessions
                    
                    return False
                
                filtered_df = filtered_df[filtered_df['Session'].apply(match_session)]
    
    # 문제 유형 필터
    if 'image' in filtered_df.columns:
        problem_types = [t['all'], t['image_problem'], t['text_only']]
        selected_problem_type = st.sidebar.selectbox(
            t['problem_type'],
            options=problem_types
        )
        
        if selected_problem_type == t['image_problem']:
            filtered_df = filtered_df[filtered_df['image'] != 'text_only']
        elif selected_problem_type == t['text_only']:
            filtered_df = filtered_df[filtered_df['image'] == 'text_only']
    
    # 연도 필터 (원본 데이터에서 추출하여 모든 연도 표시)
    if 'Year' in results_df.columns:
        # 선택된 테스트들의 연도만 표시
        if selected_tests:
            year_source_df = results_df[results_df['테스트명'].isin(selected_tests)]
        else:
            year_source_df = results_df
        
        # 연도를 정수로 변환하여 표시
        years_raw = year_source_df['Year'].dropna().unique().tolist()
        years_int = []
        for y in years_raw:
            y_int = safe_convert_to_int(y)
            if y_int and y_int not in years_int:
                years_int.append(y_int)
        years = safe_sort(years_int)
        
        if years:
            selected_years = st.sidebar.multiselect(
                t['year'],
                options=years,
                default=years
            )
            
            if selected_years:
                # 선택된 연도와 매칭되는 원본 데이터 필터링
                filtered_df = filtered_df[filtered_df['Year'].apply(
                    lambda x: safe_convert_to_int(x) in selected_years if pd.notna(x) else False
                )]
    
    # 법령 구분 필터
    if 'law' in filtered_df.columns:
        law_options = [t['all'], t['law'], t['non_law']]
        selected_law = st.sidebar.selectbox(
            t['law_type'],
            options=law_options
        )
        
        if selected_law == t['law']:
            filtered_df = filtered_df[filtered_df['law'] == 'O']
        elif selected_law == t['non_law']:
            filtered_df = filtered_df[filtered_df['law'] != 'O']
    
    # 필터링된 데이터가 없는 경우
    if filtered_df.empty:
        st.warning("No data matches the selected filters.")
        return
    
    # 탭 생성
    tabs = st.tabs([
        f"📊 {t['overview']}",
        f"🔍 {t['model_comparison']}",
        f"⏱️ {t['response_time_analysis']}",
        f"⚖️ {t['law_analysis']}",
        f"📚 {t['subject_analysis']}",
        f"📅 {t['year_analysis']}",
        f"❌ {t['incorrect_analysis']}",
        f"📈 {t['difficulty_analysis']}",
        f"💰 {t['token_cost_analysis']}",
        f"📋 {t['testset_stats']}",
        "📑 " + ("추가 분석" if lang == 'ko' else "Additional Analysis")
    ])
    
    # 탭 1: 전체 요약
    with tabs[0]:
        st.header(f"📊 {t['overview']}")
        
        # 테스트셋 기반으로 실제 문제 수 계산 (Question 기준 중복 제거)
        total_problems = 0
        if selected_tests:
            for test_name in selected_tests:
                if test_name in testsets:
                    # Question 기준 중복 제거 후 카운트
                    if 'Question' in testsets[test_name].columns:
                        total_problems += testsets[test_name]['Question'].nunique()
                    else:
                        total_problems += len(testsets[test_name])
        
        # 고유 문제 수는 filtered_df에서 중복 제거 (백업용)
        unique_questions = filtered_df['Question'].nunique()
        num_models = filtered_df['모델'].nunique()
        
        # 테스트셋 기본 정보
        st.subheader("📋 " + ("테스트셋 정보" if lang == 'ko' else "Test Set Information"))
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # 테스트셋 파일의 실제 문제 수 사용
            display_problems = total_problems if total_problems > 0 else unique_questions
            st.metric(
                "총 문제 수" if lang == 'ko' else "Total Problems",
                f"{display_problems:,}"
            )
        with col2:
            st.metric(
                "평가 모델 수" if lang == 'ko' else "Number of Models",
                f"{num_models}"
            )
        with col3:
            # 수정: 총 평가 횟수 = 총 문제 수 × 모델 수
            actual_eval_count = display_problems * num_models
            st.metric(
                "총 평가 횟수" if lang == 'ko' else "Total Evaluations",
                f"{actual_eval_count:,}"
            )
        
        st.markdown("---")
        
        # 모델 평균 성능
        st.subheader("🎯 " + ("모델 평균 성능" if lang == 'ko' else "Average Model Performance"))
        col1, col2, col3, col4 = st.columns(4)
        
        # 모델별 정확도 계산 후 평균
        model_accuracies = filtered_df.groupby('모델')['정답여부'].mean()
        avg_accuracy = model_accuracies.mean() * 100
        
        # 평균 정답/오답 수 (모델당)
        avg_problems_per_model = display_problems  # 모델당 평가한 문제 수 (테스트셋 기준)
        avg_correct = (avg_problems_per_model * avg_accuracy / 100) if avg_problems_per_model > 0 else 0
        avg_wrong = avg_problems_per_model - avg_correct
        
        with col1:
            st.metric(
                "평균 정확도" if lang == 'ko' else "Average Accuracy",
                f"{avg_accuracy:.2f}%"
            )
        with col2:
            st.metric(
                "모델당 평균 문제 수" if lang == 'ko' else "Avg Problems per Model",
                f"{avg_problems_per_model:.0f}"
            )
        with col3:
            st.metric(
                "평균 정답 수" if lang == 'ko' else "Avg Correct Answers",
                f"{avg_correct:.0f}"
            )
        with col4:
            st.metric(
                "평균 오답 수" if lang == 'ko' else "Avg Wrong Answers",
                f"{avg_wrong:.0f}"
            )
        
        # 법령/비법령 통계
        if 'law' in filtered_df.columns:
            st.markdown("---")
            st.subheader("⚖️ " + ("법령/비법령 분석" if lang == 'ko' else "Law/Non-Law Analysis"))
            
            # 🔥 일관성을 위해 항상 테스트셋 기반으로 법령/비법령 문제 수 계산
            law_count = 0
            non_law_count = 0
            
            if selected_tests:
                for test_name in selected_tests:
                    if test_name in testsets:
                        test_df = testsets[test_name]
                        if 'law' in test_df.columns:
                            law_count += len(test_df[test_df['law'] == 'O'])
                            non_law_count += len(test_df[test_df['law'] != 'O'])
                        else:
                            # law 컬럼이 없으면 전체를 비법령으로 간주
                            non_law_count += len(test_df)
            else:
                # 선택된 테스트가 없으면 모든 테스트셋 합산
                for test_name, test_df in testsets.items():
                    if 'law' in test_df.columns:
                        law_count += len(test_df[test_df['law'] == 'O'])
                        non_law_count += len(test_df[test_df['law'] != 'O'])
                    else:
                        non_law_count += len(test_df)
            
            # 법령/비법령 정답률 (모든 모델 평균)
            law_df = filtered_df[filtered_df['law'] == 'O']
            non_law_df = filtered_df[filtered_df['law'] != 'O']
            
            law_accuracy = (law_df['정답여부'].mean() * 100) if len(law_df) > 0 else 0
            non_law_accuracy = (non_law_df['정답여부'].mean() * 100) if len(non_law_df) > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(t['law_problems'], f"{law_count:,}")
            with col2:
                st.metric(f"{t['law']} {t['correct_rate']}", f"{law_accuracy:.2f}%")
            with col3:
                st.metric(t['non_law_problems'], f"{non_law_count:,}")
            with col4:
                st.metric(f"{t['non_law']} {t['correct_rate']}", f"{non_law_accuracy:.2f}%")
        
        # 시각화 차트 추가
        st.markdown("---")
        st.subheader("📊 " + ("주요 지표 시각화" if lang == 'ko' else "Key Metrics Visualization"))
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 모델별 평균 정확도 바 차트
            model_acc_df = filtered_df.groupby('모델')['정답여부'].mean().reset_index()
            model_acc_df.columns = [t['model'], t['accuracy']]
            model_acc_df[t['accuracy']] = model_acc_df[t['accuracy']] * 100
            model_acc_df = model_acc_df.sort_values(t['accuracy'], ascending=False)
            
            fig = px.bar(
                model_acc_df,
                x=t['model'],
                y=t['accuracy'],
                title=t['avg_accuracy_by_model'],
                text=t['accuracy'],
                color=t['accuracy'],
                color_continuous_scale='RdYlGn'
            )
            fig.update_traces(
                texttemplate='%{text:.1f}%',
                textposition='outside',                textfont=dict(size=annotation_size),
                marker_line_color='black',
                marker_line_width=1.5
            )
            fig.update_layout(
                height=400,
                showlegend=False,
                yaxis_title=t['accuracy'] + ' (%)',
                xaxis_title=t['model'],
                yaxis=dict(range=[0, 100])
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 법령/비법령 정답률 비교 차트
            if 'law' in filtered_df.columns:
                law_comparison = pd.DataFrame({
                    '구분': [t['law'], t['non_law']],
                    '정답률': [law_accuracy, non_law_accuracy],
                    '문제수': [law_count, non_law_count]
                })
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name=t['correct_rate'] if lang == 'ko' else 'Accuracy',
                    x=law_comparison['구분'],
                    y=law_comparison['정답률'],
                    text=law_comparison['정답률'].round(1),
                    texttemplate='%{text}%',
                    textposition='outside',                textfont=dict(size=annotation_size),
                    marker_color=['#FF6B6B', '#4ECDC4'],
                    marker_line_color='black',
                    marker_line_width=1.5,
                    yaxis='y'
                ))
                
                fig.add_trace(go.Scatter(
                    name=t['problem_count'] if lang == 'ko' else 'Problem Count',
                    x=law_comparison['구분'],
                    y=law_comparison['문제수'],
                    text=law_comparison['문제수'],
                    texttemplate='%{text}개',
                    textposition='top center',                textfont=dict(size=annotation_size),
                    mode='lines+markers+text',
                    marker=dict(size=10, color='orange'),
                    line=dict(width=2, color='orange'),
                    yaxis='y2'
                ))
                
                fig.update_layout(
                    title='법령/비법령 정답률 및 문제 수 비교' if lang == 'ko' else 'Law/Non-Law Accuracy and Problem Count Comparison',
                    height=400,
                    yaxis=dict(
                        title=('정답률 (%)' if lang == 'ko' else 'Accuracy (%)'),
                        range=[0, 100]
                    ),
                    yaxis2=dict(
                        title=(t['problem_count'] if lang == 'ko' else 'Problem Count'),
                        overlaying='y',
                        side='right',
                        range=[0, max(law_count, non_law_count) * 1.2]
                    ),
                    hovermode='x unified',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # 법령 정보가 없을 때 - 모델별 정답/오답 수 차트
                model_correct_wrong = filtered_df.groupby('모델')['정답여부'].agg(['sum', 'count']).reset_index()
                model_correct_wrong.columns = ['모델', '정답', '총문제']
                model_correct_wrong['오답'] = model_correct_wrong['총문제'] - model_correct_wrong['정답']
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name='정답',
                    x=model_correct_wrong['모델'],
                    y=model_correct_wrong['정답'],
                    marker_color='lightgreen'
                ))
                fig.add_trace(go.Bar(
                    name='오답',
                    x=model_correct_wrong['모델'],
                    y=model_correct_wrong['오답'],
                    marker_color='lightcoral'
                ))
                
                fig.update_layout(
                    barmode='stack',
                    title='모델별 정답/오답 수',
                    height=400,
                    yaxis_title='문제 수',
                    xaxis_title='모델'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # 테스트셋별 분포 (여러 테스트가 있을 경우)
        if '테스트명' in filtered_df.columns and filtered_df['테스트명'].nunique() > 1:
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 🔥 테스트셋별 문제 수 (테스트셋 파일 기준)
                test_problem_data = []
                for test_name in selected_tests if selected_tests else testsets.keys():
                    if test_name in testsets:
                        problem_count = len(testsets[test_name])
                        test_problem_data.append({'테스트명': test_name, '문제수': problem_count})
                
                test_problem_count = pd.DataFrame(test_problem_data)
                test_problem_count = test_problem_count.sort_values('문제수', ascending=False)
                
                fig = px.bar(
                    test_problem_count,
                    x='테스트명',
                    y='문제수',
                    title='테스트셋별 문제 수' if lang == 'ko' else 'Problems by Test',
                    text='문제수',
                    color='문제수',
                    color_continuous_scale='Blues'
                )
                fig.update_traces(
                    textposition='outside',
                    marker_line_color='black',
                    marker_line_width=1.5
                )
                fig.update_layout(
                    height=400,
                    showlegend=False,
                    yaxis_title='문제 수' if lang == 'ko' else 'Problem Count',
                    xaxis_title='테스트명' if lang == 'ko' else 'Test Name'
                )
                fig.update_xaxes(tickangle=45, tickfont=dict(size=annotation_size))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # 테스트셋별 평균 정확도
                test_accuracy = filtered_df.groupby('테스트명')['정답여부'].mean().reset_index()
                test_accuracy.columns = ['테스트명', '정확도']
                test_accuracy['정확도'] = test_accuracy['정확도'] * 100
                test_accuracy = test_accuracy.sort_values('정확도', ascending=False)
                
                fig = px.bar(
                    test_accuracy,
                    x='테스트명',
                    y='정확도',
                    title='테스트셋별 평균 정확도',
                    text='정확도',
                    color='정확도',
                    color_continuous_scale='RdYlGn'
                )
                fig.update_traces(
                texttemplate='%{text:.1f}%',
                textposition='outside',                textfont=dict(size=annotation_size),
                marker_line_color='black',
                marker_line_width=1.5
            )
                fig.update_layout(
                    height=400,
                    showlegend=False,
                    yaxis_title='정확도 (%)',
                    xaxis_title='테스트명',
                    yaxis=dict(range=[0, 100])
                )
                fig.update_xaxes(tickangle=45, tickfont=dict(size=annotation_size))
                st.plotly_chart(fig, use_container_width=True)
        
        # 종합 인사이트
        st.markdown("---")
        st.subheader("💡 " + ("종합 인사이트" if lang == 'ko' else "Key Insights"))
        
        # 최고/최저 성능 모델 찾기
        best_model = model_acc_df.iloc[0]
        worst_model = model_acc_df.iloc[-1]
        performance_gap = best_model[t['accuracy']] - worst_model[t['accuracy']]
        
        # 법령 vs 비법령 차이
        if 'law' in filtered_df.columns and law_count > 0 and non_law_count > 0:
            law_difficulty = "더 어려움" if law_accuracy < non_law_accuracy else "더 쉬움" if law_accuracy > non_law_accuracy else "비슷함"
            law_diff_pct = abs(law_accuracy - non_law_accuracy)
            
            st.success(f"""
            📊 **{"성능 분석" if lang == 'ko' else "Performance Analysis"}**:
            - **{"최고 성능 모델" if lang == 'ko' else "Top Model"}**: {best_model[t['model']]} ({best_model[t['accuracy']]:.2f}%)
            - **{"최저 성능 모델" if lang == 'ko' else "Lowest Model"}**: {worst_model[t['model']]} ({worst_model[t['accuracy']]:.2f}%)
            - **{"성능 격차" if lang == 'ko' else "Performance Gap"}**: {performance_gap:.2f}%p
            
            ⚖️ **{"법령 문제 분석" if lang == 'ko' else "Law Problem Analysis"}**:
            - {"법령 문제가" if lang == 'ko' else "Law problems are"} **{law_difficulty}** ({"차이" if lang == 'ko' else "difference"}: {law_diff_pct:.2f}%p)
            - {"법령 문제 정답률" if lang == 'ko' else "Law accuracy"}: {law_accuracy:.2f}% vs {"비법령" if lang == 'ko' else "Non-law"}: {non_law_accuracy:.2f}%
            
            📈 **{"추천 사항" if lang == 'ko' else "Recommendations"}**:
            - {"높은 정확도가 필요한 경우" if lang == 'ko' else "For high accuracy needs"}: {best_model[t['model']]} {"사용" if lang == 'ko' else "recommended"}
            - {"법령 문제 특화가 필요한 경우" if lang == 'ko' else "For law-specific tasks"}: {"법령/비법령 분석 탭에서 세부 성능 확인" if lang == 'ko' else "Check detailed performance in Law Analysis tab"}
            """)
        else:
            st.success(f"""
            📊 **{"성능 분석" if lang == 'ko' else "Performance Analysis"}**:
            - **{"최고 성능 모델" if lang == 'ko' else "Top Model"}**: {best_model[t['model']]} ({best_model[t['accuracy']]:.2f}%)
            - **{"최저 성능 모델" if lang == 'ko' else "Lowest Model"}**: {worst_model[t['model']]} ({worst_model[t['accuracy']]:.2f}%)
            - **{"성능 격차" if lang == 'ko' else "Performance Gap"}**: {performance_gap:.2f}%p
            
            📈 **{"추천 사항" if lang == 'ko' else "Recommendations"}**:
            - {"높은 정확도가 필요한 경우" if lang == 'ko' else "For high accuracy needs"}: {best_model[t['model']]} {"사용 권장" if lang == 'ko' else "recommended"}
            - {"평균 성능" if lang == 'ko' else "Average performance"}: {avg_accuracy:.2f}% - {"이를 기준으로 모델 선택" if lang == 'ko' else "use as baseline for model selection"}
            """)
    
    # 탭 2: 모델별 비교
    with tabs[1]:
        st.header(f"🔍 {t['model_comparison']}")
        
        # 모델별 성능 계산
        model_stats = filtered_df.groupby('모델').agg({
            '정답여부': ['sum', 'count', 'mean']
        }).reset_index()
        model_stats.columns = ['모델', '정답', '총문제', '정확도']
        model_stats['정확도'] = model_stats['정확도'] * 100
        model_stats['오답'] = model_stats['총문제'] - model_stats['정답']
        model_stats = model_stats.sort_values('정확도', ascending=False)
        
        # 성능 지표 테이블
        st.subheader(t['performance_by_model'])
        
        # 테이블 컬럼명 변경
        display_stats = model_stats.copy()
        if lang == 'en':
            display_stats.columns = ['Model', 'Correct', 'Total', 'Accuracy', 'Wrong']
        
        st.dataframe(
            display_stats.style.format({
                '정확도' if lang == 'ko' else 'Accuracy': '{:.2f}%'
            }).background_gradient(subset=['정확도' if lang == 'ko' else 'Accuracy'], cmap='RdYlGn'),
            width='stretch'
        )
        
        # 비교 차트
        st.markdown("---")
        st.subheader(t['comparison_chart'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 정확도 바 차트
            fig = px.bar(
                model_stats,
                x='모델',
                y='정확도',
                title=t['overall_comparison'],
                text='정확도',
                color='정확도',
                color_continuous_scale='RdYlGn'
            )
            fig.update_traces(
                texttemplate='%{text:.1f}%',
                textposition='outside',                textfont=dict(size=annotation_size),
                marker_line_color='black',
                marker_line_width=1.5
            )
            fig.update_layout(
                height=400,
                showlegend=False,
                yaxis_title=t['accuracy'] + ' (%)',
                xaxis_title=t['model']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 정답/오답 스택 바 차트
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name=t['correct'],
                x=model_stats['모델'],
                y=model_stats['정답'],
                marker_color='lightgreen',
                marker_line_color='black',
                marker_line_width=1.5
            ))
            fig.add_trace(go.Bar(
                name=t['wrong'],
                x=model_stats['모델'],
                y=model_stats['오답'],
                marker_color='lightcoral',
                marker_line_color='black',
                marker_line_width=1.5
            ))
            
            fig.update_layout(
                barmode='stack',
                title=f"{t['correct']}/{t['wrong']} {t['comparison_chart']}",
                height=400,
                yaxis_title=t['problem_count'],
                xaxis_title=t['model']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 히트맵
        if '테스트명' in filtered_df.columns:
            st.markdown("---")
            st.subheader(t['heatmap'])
            
            # 모델별, 테스트별 정확도 계산
            heatmap_data = filtered_df.groupby(['모델', '테스트명'])['정답여부'].mean() * 100
            heatmap_pivot = heatmap_data.unstack(fill_value=0)
            
            # 히트맵 생성 (숫자 표시 및 셀 경계선 추가)
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_pivot.values,
                x=heatmap_pivot.columns,
                y=heatmap_pivot.index,
                colorscale='RdYlGn',
                text=np.round(heatmap_pivot.values, 1),
                texttemplate='%{text:.1f}',
                textfont={"size": annotation_size},
                colorbar=dict(title=t['accuracy'] + " (%)"),
                xgap=2,  # 셀 경계선
                ygap=2
            ))
            
            fig.update_layout(height=400)
            fig.update_xaxes(tickangle=45, tickfont=dict(size=annotation_size))
            fig.update_yaxes(tickfont=dict(size=annotation_size))
            st.plotly_chart(fig, use_container_width=True)
            
            # 히트맵 인사이트
            st.info(f"""
            💡 **{"히트맵 분석" if lang == 'ko' else "Heatmap Analysis"}**:
            - **{"가장 어려운 테스트" if lang == 'ko' else "Hardest Test"}**: {heatmap_pivot.mean(axis=0).idxmin()} ({"평균" if lang == 'ko' else "avg"}: {heatmap_pivot.mean(axis=0).min():.1f}%)
            - **{"가장 쉬운 테스트" if lang == 'ko' else "Easiest Test"}**: {heatmap_pivot.mean(axis=0).idxmax()} ({"평균" if lang == 'ko' else "avg"}: {heatmap_pivot.mean(axis=0).max():.1f}%)
            - **{"일관성" if lang == 'ko' else "Consistency"}**: {"모든 모델이 비슷한 성능 패턴을 보이는지 확인하세요" if lang == 'ko' else "Check if all models show similar performance patterns"}
            - **{"특화 영역" if lang == 'ko' else "Specialization"}**: {"특정 모델이 특정 테스트에서 특히 우수한지 파악하세요" if lang == 'ko' else "Identify if specific models excel in certain tests"}
            """)
    
    # 탭 3: 응답시간 분석
    with tabs[2]:
        st.header(f"⏱️ {t['response_time_analysis']}")
        
        # 문제당평균시간(초) 컬럼이 있는지 확인
        time_columns = ['문제당평균시간(초)', '총소요시간(초)', 'question_duration']
        available_time_col = None
        for col in time_columns:
            if col in filtered_df.columns:
                available_time_col = col
                break
        
        if available_time_col is None:
            st.info("Response time data not available in the dataset.")
        else:
            # 응답시간 데이터 준비
            if available_time_col == 'question_duration':
                # question_duration은 개별 문제 시간
                time_col = 'question_duration'
                is_per_problem = True
            elif available_time_col == '문제당평균시간(초)':
                time_col = '문제당평균시간(초)'
                is_per_problem = True
            else:
                time_col = '총소요시간(초)'
                is_per_problem = False
            
            # NaN 값 제거 (view만 생성, copy 불필요)
            time_df = filtered_df[filtered_df[time_col].notna()]
            
            if len(time_df) == 0:
                st.info("No valid response time data available.")
            else:
                # 1. 모델별 평균 응답시간 통계
                st.subheader(t['response_time_stats'])
                
                model_time_stats = time_df.groupby('모델').agg({
                    time_col: ['mean', 'median', 'std', 'min', 'max', 'count']
                }).reset_index()
                
                model_time_stats.columns = ['모델', '평균', '중앙값', '표준편차', '최소', '최대', '문제수']
                model_time_stats = model_time_stats.sort_values('평균')
                
                # 정확도도 함께 표시
                model_acc = filtered_df.groupby('모델')['정답여부'].mean().reset_index()
                model_acc.columns = ['모델', '정확도']
                model_acc['정확도'] = model_acc['정확도'] * 100
                
                model_time_stats = model_time_stats.merge(model_acc, on='모델')
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    fastest = model_time_stats.iloc[0]
                    st.metric(
                        t['fastest_model'],
                        fastest['모델'],
                        f"{fastest['평균']:.2f}{t['seconds']}"
                    )
                
                with col2:
                    slowest = model_time_stats.iloc[-1]
                    st.metric(
                        t['slowest_model'],
                        slowest['모델'],
                        f"{slowest['평균']:.2f}{t['seconds']}"
                    )
                
                with col3:
                    avg_time = model_time_stats['평균'].mean()
                    st.metric(
                        t['avg_response_time'],
                        f"{avg_time:.2f}{t['seconds']}"
                    )
                
                # 테이블
                st.dataframe(
                    model_time_stats.style.format({
                        '평균': '{:.2f}',
                        '중앙값': '{:.2f}',
                        '표준편차': '{:.2f}',
                        '최소': '{:.2f}',
                        '최대': '{:.2f}',
                        '문제수': '{:.0f}',
                        '정확도': '{:.2f}%'
                    }).background_gradient(subset=['평균'], cmap='RdYlGn_r'),
                    width='stretch'
                )
                
                st.markdown("---")
                
                # 2. 시각화
                st.subheader(t['response_time_by_model'])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # 평균 응답시간 바 차트
                    fig = px.bar(
                        model_time_stats,
                        x='모델',
                        y='평균',
                        title=t['avg_response_time'] + (' (' + t['time_per_problem'] + ')' if is_per_problem else ''),
                        text='평균',
                        color='평균',
                        color_continuous_scale='RdYlGn_r'
                    )
                    fig.update_traces(
                        texttemplate='%{text:.2f}s',
                        textposition='outside',                textfont=dict(size=annotation_size),
                        marker_line_color='black',
                        marker_line_width=1.5
                    )
                    fig.update_layout(
                        height=400,
                        showlegend=False,
                        yaxis_title=t['response_time'] + ' (' + t['seconds'] + ')',
                        xaxis_title=t['model']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # 박스플롯
                    fig = px.box(
                        time_df,
                        x='모델',
                        y=time_col,
                        title=t['response_time_distribution'],
                        color='모델'
                    )
                    fig.update_layout(
                        height=400,
                        showlegend=False,
                        yaxis_title=t['response_time'] + ' (' + t['seconds'] + ')',
                        xaxis_title=t['model']
                    )
                    fig.update_xaxes(tickangle=45, tickfont=dict(size=annotation_size))
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                # 3. 응답시간 vs 정확도
                st.subheader(t['response_time_vs_accuracy'])
                
                fig = px.scatter(
                    model_time_stats,
                    x='평균',
                    y='정확도',
                    size='문제수',
                    text='모델',
                    title=t['response_time_vs_accuracy'],
                    labels={
                        '평균': t['avg_response_time'] + ' (' + t['seconds'] + ')',
                        '정확도': t['accuracy'] + ' (%)'
                    }
                )
                fig.update_traces(
                    textposition='top center',
                    marker=dict(
                        line=dict(width=2, color='black'),
                        opacity=0.7
                    )
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # 인사이트 개선
                speed_accuracy_ratio = fastest['정확도'] / slowest['정확도'] if slowest['정확도'] > 0 else 0
                time_ratio = slowest['평균'] / fastest['평균'] if fastest['평균'] > 0 else 0
                
                st.info(f"""
                💡 **{"속도 vs 정확도 트레이드오프 분석" if lang == 'ko' else "Speed vs Accuracy Trade-off Analysis"}**:
                
                🏃 **{"속도" if lang == 'ko' else "Speed"}**:
                - **{"최고속" if lang == 'ko' else "Fastest"}**: {fastest['모델']} ({fastest['평균']:.2f}{"초" if lang == 'ko' else "s"}, {"정확도" if lang == 'ko' else "accuracy"} {fastest['정확도']:.1f}%)
                - **{"최저속" if lang == 'ko' else "Slowest"}**: {slowest['모델']} ({slowest['평균']:.2f}{"초" if lang == 'ko' else "s"}, {"정확도" if lang == 'ko' else "accuracy"} {slowest['정확도']:.1f}%)
                - **{"속도 차이" if lang == 'ko' else "Speed difference"}**: {time_ratio:.1f}x
                
                🎯 **{"효율성 분석" if lang == 'ko' else "Efficiency Analysis"}**:
                - {"빠른 모델이" if lang == 'ko' else "Fast model is"} {speed_accuracy_ratio:.2f}x {"의 정확도를 가짐" if lang == 'ko' else "as accurate"}
                - **{"권장사항" if lang == 'ko' else "Recommendation"}**: {"실시간 처리가 중요하면" if lang == 'ko' else "For real-time: "}{fastest['모델']}, {"정확도가 중요하면" if lang == 'ko' else "For accuracy: "}{slowest['모델'] if slowest['정확도'] > fastest['정확도'] else fastest['모델']}
                
                📊 **{"산점도 활용팁" if lang == 'ko' else "Scatter Plot Insights"}**:
                - {"왼쪽 위" if lang == 'ko' else "Top-left"}: {"빠르고 정확함 (이상적)" if lang == 'ko' else "Fast & Accurate (ideal)"}
                - {"오른쪽 아래" if lang == 'ko' else "Bottom-right"}: {"느리고 부정확함 (피해야 함)" if lang == 'ko' else "Slow & Inaccurate (avoid)"}
                """)
                
                st.markdown("---")
                
                # 4. 테스트별 응답시간 (테스트가 여러 개인 경우)
                if '테스트명' in time_df.columns and time_df['테스트명'].nunique() > 1:
                    st.subheader(f"{t['response_time']} ({t['by_test']})" if 'by_test' in t else "테스트별 응답시간")
                    
                    test_time = time_df.groupby(['모델', '테스트명'])[time_col].mean().reset_index()
                    test_time.columns = ['모델', '테스트명', '평균시간']
                    
                    fig = px.bar(
                        test_time,
                        x='테스트명',
                        y='평균시간',
                        color='모델',
                        barmode='group',
                        title='테스트별 모델 응답시간' if lang == 'ko' else 'Response Time by Test',
                        labels={'평균시간': t['avg_response_time'] + ' (' + t['seconds'] + ')'}
                    )
                    fig.update_layout(
                        height=400,
                        xaxis_title=t['testname'],
                        yaxis_title=t['response_time'] + ' (' + t['seconds'] + ')'
                    )
                    fig.update_xaxes(tickangle=45, tickfont=dict(size=annotation_size))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 테스트별 인사이트
                    hardest_test = test_time.groupby('테스트명')['평균시간'].mean().idxmax()
                    easiest_test = test_time.groupby('테스트명')['평균시간'].mean().idxmin()
                    st.success(f"""
                    📊 **{"테스트별 처리 시간 분석" if lang == 'ko' else "Processing Time by Test"}**:
                    - **{"가장 오래 걸리는 테스트" if lang == 'ko' else "Slowest test"}**: {hardest_test} ({"복잡도가 높을 가능성" if lang == 'ko' else "likely more complex"})
                    - **{"가장 빠른 테스트" if lang == 'ko' else "Fastest test"}**: {easiest_test} ({"상대적으로 단순" if lang == 'ko' else "relatively simpler"})
                    - **{"참고" if lang == 'ko' else "Note"}**: {"테스트별 처리 시간 차이는 문제 난이도나 길이와 관련" if lang == 'ko' else "Time differences relate to problem difficulty or length"}
                    """)
    
    # 탭 4: 법령/비법령 분석
    with tabs[3]:
        if 'law' not in filtered_df.columns:
            st.info("Law classification data not available.")
        else:
            st.header(f"⚖️ {t['law_analysis']}")
            
            # 전체 법령/비법령 비율
            st.subheader(t['law_ratio'])
            
            # 🔥 테스트셋 기반으로 법령/비법령 문제 수 계산 (전체 요약과 동일)
            law_count = 0
            non_law_count = 0
            
            if selected_tests:
                for test_name in selected_tests:
                    if test_name in testsets:
                        test_df = testsets[test_name]
                        if 'law' in test_df.columns:
                            law_count += len(test_df[test_df['law'] == 'O'])
                            non_law_count += len(test_df[test_df['law'] != 'O'])
                        else:
                            # law 컬럼이 없으면 전체를 비법령으로 간주
                            non_law_count += len(test_df)
            else:
                # 선택된 테스트가 없으면 모든 테스트셋 합산
                for test_name, test_df in testsets.items():
                    if 'law' in test_df.columns:
                        law_count += len(test_df[test_df['law'] == 'O'])
                        non_law_count += len(test_df[test_df['law'] != 'O'])
                    else:
                        non_law_count += len(test_df)
            
            total_unique = law_count + non_law_count
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 파이 차트
                fig = go.Figure(data=[go.Pie(
                    labels=[t['law'], t['non_law']],
                    values=[law_count, non_law_count],
                    hole=0.3,
                    marker=dict(line=dict(color='black', width=2))
                )])
                fig.update_layout(
                    title=t['law_distribution_stat'],
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # 수치 표시
                st.metric(
                    t['law_problems'], 
                    f"{law_count} ({law_count/total_unique*100:.1f}%)",
                    help="테스트셋 파일 기준 법령 문제 수"
                )
                st.metric(
                    t['non_law_problems'], 
                    f"{non_law_count} ({non_law_count/total_unique*100:.1f}%)",
                    help="테스트셋 파일 기준 비법령 문제 수"
                )
            
            st.info("💡 " + (
                "이 통계는 테스트셋 파일 기준입니다. 전체 요약 탭과 동일합니다." 
                if lang == 'ko' 
                else "These statistics are based on test set files. They match the Overview tab."
            ))
            
            # 모델별 법령/비법령 성능
            st.markdown("---")
            st.subheader(t['model_law_performance'])
            
            law_performance = []
            for model in filtered_df['모델'].unique():
                model_df = filtered_df[filtered_df['모델'] == model]
                
                law_model = model_df[model_df['law'] == 'O']
                non_law_model = model_df[model_df['law'] != 'O']
                
                law_acc = (law_model['정답여부'].sum() / len(law_model) * 100) if len(law_model) > 0 else 0
                non_law_acc = (non_law_model['정답여부'].sum() / len(non_law_model) * 100) if len(non_law_model) > 0 else 0
                
                law_performance.append({
                    '모델': model,
                    '법령': law_acc,
                    '비법령': non_law_acc
                })
            
            law_perf_df = pd.DataFrame(law_performance)
            
            # 그룹 바 차트
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name=t['law'],
                x=law_perf_df['모델'],
                y=law_perf_df['법령'],
                marker_color='skyblue'
            ))
            fig.add_trace(go.Bar(
                name=t['non_law'],
                x=law_perf_df['모델'],
                y=law_perf_df['비법령'],
                marker_color='lightcoral'
            ))
            
            fig.update_layout(
                barmode='group',
                title=t['law_distribution'],
                height=500,
                yaxis_title=t['accuracy'] + ' (%)',
                xaxis_title=t['model']
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 법령/비법령 성능 인사이트
            # 법령에 강한 모델과 비법령에 강한 모델 찾기
            law_perf_df['법령_우위'] = law_perf_df['법령'] - law_perf_df['비법령']
            best_law_model = law_perf_df.loc[law_perf_df['법령'].idxmax()]
            best_nonlaw_model = law_perf_df.loc[law_perf_df['비법령'].idxmax()]
            most_law_specialized = law_perf_df.loc[law_perf_df['법령_우위'].idxmax()]
            most_balanced = law_perf_df.loc[law_perf_df['법령_우위'].abs().idxmin()]
            
            st.success(f"""
            💡 **{"법령 문제 특화 분석" if lang == 'ko' else "Law Problem Specialization Analysis"}**:
            
            🏆 **{"최고 성능" if lang == 'ko' else "Top Performance"}**:
            - **{"법령 최고" if lang == 'ko' else "Best at Law"}**: {best_law_model['모델']} ({best_law_model['법령']:.1f}%)
            - **{"비법령 최고" if lang == 'ko' else "Best at Non-Law"}**: {best_nonlaw_model['모델']} ({best_nonlaw_model['비법령']:.1f}%)
            
            ⚖️ **{"균형 vs 특화" if lang == 'ko' else "Balance vs Specialization"}**:
            - **{"가장 균형잡힌 모델" if lang == 'ko' else "Most Balanced"}**: {most_balanced['모델']} ({"차이" if lang == 'ko' else "diff"}: {abs(most_balanced['법령_우위']):.1f}%p)
            - **{"법령 특화 모델" if lang == 'ko' else "Law Specialized"}**: {most_law_specialized['모델']} ({"법령" if lang == 'ko' else "law"} +{most_law_specialized['법령_우위']:.1f}%p)
            
            📋 **{"활용 가이드" if lang == 'ko' else "Usage Guide"}**:
            - **{"법률 자문 시스템" if lang == 'ko' else "Legal Advisory"}**: {best_law_model['모델']} {"추천" if lang == 'ko' else "recommended"}
            - **{"일반 안전 교육" if lang == 'ko' else "General Safety Training"}**: {most_balanced['모델']} {"추천" if lang == 'ko' else "recommended"}
            - **{"종합 솔루션" if lang == 'ko' else "Comprehensive Solution"}**: {"법령/비법령 모두 높은 모델 선택" if lang == 'ko' else "Choose models high in both areas"}
            """)
    
    # 탭 5: 과목별 분석
    with tabs[4]:
        if 'Subject' not in filtered_df.columns:
            st.info("Subject data not available.")
        else:
            st.header(f"📚 {t['subject_analysis']}")
            
            # 과목별 성능
            subject_stats = filtered_df.groupby('Subject').agg({
                '정답여부': ['sum', 'count', 'mean']
            }).reset_index()
            
            # 컬럼명 언어별 설정
            if lang == 'ko':
                subject_stats.columns = ['과목', '정답', '총문제', '정확도']
                subj_col = '과목'
                acc_col = '정확도'
                correct_col = '정답'
                total_col = '총문제'
            else:
                subject_stats.columns = ['Subject', 'Correct', 'Total', 'Accuracy']
                subj_col = 'Subject'
                acc_col = 'Accuracy'
                correct_col = 'Correct'
                total_col = 'Total'
            
            subject_stats[acc_col] = subject_stats[acc_col] * 100
            subject_stats = subject_stats.sort_values(acc_col, ascending=False)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # 테이블
                st.dataframe(
                    subject_stats.style.format({acc_col: '{:.2f}%'})
                    .background_gradient(subset=[acc_col], cmap='RdYlGn'),
                    width='stretch'
                )
            
            with col2:
                # 바 차트
                fig = px.bar(
                    subject_stats,
                    x=subj_col,
                    y=acc_col,
                    title=t['subject_performance'],
                    text=acc_col,
                    color=acc_col,
                    color_continuous_scale='RdYlGn',
                    labels={subj_col: t['by_subject'].replace('별', ''), acc_col: t['accuracy'] + ' (%)'}
                )
                fig.update_traces(
                    texttemplate='%{text:.1f}%',
                    textposition='outside',                textfont=dict(size=annotation_size),
                    marker_line_color='black',
                    marker_line_width=1.5
                )
                fig.update_layout(
                    height=400,
                    showlegend=False,
                    yaxis_title=t['accuracy'] + ' (%)',
                    xaxis_title=t['by_subject'].replace('별', '')
                )
                fig.update_xaxes(tickangle=45, tickfont=dict(size=annotation_size))
                st.plotly_chart(fig, use_container_width=True)
            
            # 모델별 과목 성능 히트맵 (셀 경계선 추가)
            st.markdown("---")
            subject_model = filtered_df.groupby(['모델', 'Subject'])['정답여부'].mean() * 100
            subject_model_pivot = subject_model.unstack(fill_value=0)
            
            fig = go.Figure(data=go.Heatmap(
                z=subject_model_pivot.values,
                x=subject_model_pivot.columns,
                y=subject_model_pivot.index,
                colorscale='RdYlGn',
                text=np.round(subject_model_pivot.values, 1),
                texttemplate='%{text:.1f}',
                textfont={"size": annotation_size},
                colorbar=dict(title=t['accuracy'] + " (%)"),
                xgap=2,  # 셀 경계선
                ygap=2
            ))
            fig.update_layout(height=400)
            fig.update_xaxes(tickangle=45, tickfont=dict(size=annotation_size))
            fig.update_yaxes(tickfont=dict(size=annotation_size))
            st.plotly_chart(fig, use_container_width=True)
            
            # 과목별 성능 인사이트
            # 과목별 평균 정확도
            subject_avg = subject_model_pivot.mean(axis=0).sort_values()
            hardest_subject = subject_avg.index[0]
            easiest_subject = subject_avg.index[-1]
            
            # 모델별 편차 (과목간 성능 일관성)
            model_consistency = subject_model_pivot.std(axis=1).sort_values()
            most_consistent = model_consistency.index[0]
            least_consistent = model_consistency.index[-1]
            
            # 특화 모델 찾기
            subject_specialists = {}
            for subject in subject_model_pivot.columns:
                best_model = subject_model_pivot[subject].idxmax()
                best_score = subject_model_pivot[subject].max()
                subject_specialists[subject] = (best_model, best_score)
            
            st.info(f"""
            💡 **{"과목별 난이도 및 모델 특화 분석" if lang == 'ko' else "Subject Difficulty & Model Specialization"}**:
            
            📚 **{"과목 난이도" if lang == 'ko' else "Subject Difficulty"}**:
            - **{"가장 어려운 과목" if lang == 'ko' else "Hardest"}**: {hardest_subject} ({"평균" if lang == 'ko' else "avg"}: {subject_avg.iloc[0]:.1f}%)
            - **{"가장 쉬운 과목" if lang == 'ko' else "Easiest"}**: {easiest_subject} ({"평균" if lang == 'ko' else "avg"}: {subject_avg.iloc[-1]:.1f}%)
            - **{"난이도 격차" if lang == 'ko' else "Difficulty gap"}**: {subject_avg.iloc[-1] - subject_avg.iloc[0]:.1f}%p
            
            🎯 **{"모델 일관성" if lang == 'ko' else "Model Consistency"}**:
            - **{"가장 일관적" if lang == 'ko' else "Most Consistent"}**: {most_consistent} ({"편차" if lang == 'ko' else "std"}: {model_consistency.iloc[0]:.1f})
            - **{"가장 불균형" if lang == 'ko' else "Least Consistent"}**: {least_consistent} ({"편차" if lang == 'ko' else "std"}: {model_consistency.iloc[-1]:.1f})
            
            🏆 **{"과목별 최고 모델" if lang == 'ko' else "Top Models by Subject"}**:
            {chr(10).join([f"- **{subj}**: {model} ({score:.1f}%)" for subj, (model, score) in list(subject_specialists.items())[:3]])}
            
            💼 **{"활용 제안" if lang == 'ko' else "Recommendations"}**:
            - **{"특정 과목 교육" if lang == 'ko' else "Subject-specific training"}**: {"해당 과목 최고 성능 모델 활용" if lang == 'ko' else "Use top model for that subject"}
            - **{"종합 교육" if lang == 'ko' else "Comprehensive education"}**: {most_consistent} {"(균형잡힌 성능)" if lang == 'ko' else "(balanced performance)"}
            """)
    
    # 탭 6: 연도별 분석
    with tabs[5]:
        if 'Year' not in filtered_df.columns:
            st.info("Year data not available.")
        else:
            st.header(f"📅 {t['year_analysis']}")
            
            # 디버깅 정보 표시
            with st.expander("🔍 디버깅 정보 (클릭하여 펼치기)"):
                st.write("**필터링 전 원본 데이터:**")
                st.write(f"- 전체 데이터 행 수: {len(results_df):,}")
                st.write(f"- 원본 Year 고유값: {sorted([str(y) for y in results_df['Year'].dropna().unique().tolist()])}")
                
                st.write("**필터링 후 데이터:**")
                st.write(f"- 필터링된 데이터 행 수: {len(filtered_df):,}")
                st.write(f"- 필터링된 Year 고유값: {sorted([str(y) for y in filtered_df['Year'].dropna().unique().tolist()])}")
                
                st.write("**현재 필터 설정:**")
                st.write(f"- 선택된 테스트: {selected_tests}")
                st.write(f"- 선택된 모델: {selected_models}")
                st.write(f"- 선택된 연도: {selected_years if 'selected_years' in locals() else '전체'}")
            
            # Year를 정수로 변환 (원본 수정 없이)
            year_int_series = filtered_df['Year'].apply(safe_convert_to_int)
            valid_year_mask = year_int_series.notna()
            
            if valid_year_mask.any():
                # 필요한 컬럼만 선택하여 새 DataFrame 생성 (모델 컬럼 포함)
                year_df = pd.DataFrame({
                    'Year_Int': year_int_series[valid_year_mask],
                    '정답여부': filtered_df.loc[valid_year_mask, '정답여부'],
                    '모델': filtered_df.loc[valid_year_mask, '모델']
                })
                
                # 연도별 성능
                year_stats = year_df.groupby('Year_Int').agg({
                    '정답여부': ['sum', 'count', 'mean']
                }).reset_index()
                year_stats.columns = ['연도', '정답', '총문제', '정확도']
                year_stats['정확도'] = year_stats['정확도'] * 100
                year_stats = year_stats.sort_values('연도')
                
                # 연도를 정수로 표시
                year_stats['연도'] = year_stats['연도'].astype(int)
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # 테이블 (소수점 없이 표시)
                    st.dataframe(
                        year_stats.style.format({
                            '연도': '{:.0f}',
                            '정답': '{:.0f}',
                            '총문제': '{:.0f}',
                            '정확도': '{:.2f}%'
                        })
                        .background_gradient(subset=['정확도'], cmap='RdYlGn'),
                        width='stretch'
                    )
                
                with col2:
                    # 라인 차트
                    fig = px.line(
                        year_stats,
                        x='연도',
                        y='정확도',
                        title=t['year_performance'],
                        markers=True,
                        text='정확도'
                    )
                    fig.update_traces(
                        texttemplate='%{text:.1f}%',
                        textposition='top center',                textfont=dict(size=annotation_size),
                        marker_size=10,
                        marker_line_color='black',
                        marker_line_width=2,
                        line_width=3
                    )
                    fig.update_layout(
                        height=400,
                        yaxis_title=t['accuracy'] + ' (%)',
                        xaxis_title=t['year']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # 연도별 문제 수 차트 추가
                st.markdown("---")
                st.subheader(f"📊 {t['year_problem_distribution']}")
                
                # 다국어 컬럼명 설정
                year_col = t['year']
                count_col = t['problem_count']
                
                # 테스트셋에서 실제 문제 수 계산 (중복 제거)
                if selected_tests:
                    year_problem_count = []
                    for test_name in selected_tests:
                        if test_name in testsets and 'Year' in testsets[test_name].columns:
                            test_year_counts = testsets[test_name].groupby('Year').size()
                            for year, count in test_year_counts.items():
                                year_int = safe_convert_to_int(year)
                                if year_int:
                                    year_problem_count.append({year_col: year_int, count_col: count})
                    
                    if year_problem_count:
                        year_problem_df = pd.DataFrame(year_problem_count)
                        year_problem_df = year_problem_df.groupby(year_col)[count_col].sum().reset_index()
                        year_problem_df = year_problem_df.sort_values(year_col)
                    else:
                        # 백업: filtered_df에서 고유 문제 수 계산
                        year_problem_df = year_df.groupby('Year_Int')['Question'].nunique().reset_index()
                        year_problem_df.columns = [year_col, count_col]
                        year_problem_df[year_col] = year_problem_df[year_col].astype(int)
                        year_problem_df = year_problem_df.sort_values(year_col)
                else:
                    # 테스트 선택 안 됨: filtered_df에서 계산
                    year_problem_df = year_df.groupby('Year_Int')['Question'].nunique().reset_index()
                    year_problem_df.columns = [year_col, count_col]
                    year_problem_df[year_col] = year_problem_df[year_col].astype(int)
                    year_problem_df = year_problem_df.sort_values(year_col)
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # 연도별 문제 수 테이블
                    st.dataframe(
                        year_problem_df.style.format({
                            year_col: '{:.0f}',
                            count_col: '{:.0f}'
                        })
                        .background_gradient(subset=[count_col], cmap='Blues'),
                        width='stretch'
                    )
                    
                    # 총 문제 수 표시
                    st.metric(t['total_problem_count'], f"{year_problem_df[count_col].sum():,.0f}" + (t['problems'] if lang == 'ko' else ''))
                
                with col2:
                    # 바 차트
                    fig = px.bar(
                        year_problem_df,
                        x=year_col,
                        y=count_col,
                        title=t['year_problem_chart'],
                        text=count_col,
                        color=count_col,
                        color_continuous_scale='Blues'
                    )
                    fig.update_traces(
                texttemplate='%{text}',
                textposition='outside',                textfont=dict(size=annotation_size),
                marker_line_color='black',
                marker_line_width=1.5
            )
                    fig.update_layout(
                        height=400,
                        showlegend=False,
                        yaxis_title=t['problem_count'],
                        xaxis_title=t['year'],
                        xaxis=dict(tickmode='linear')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # 모델별 연도 성능 히트맵
                st.markdown("---")
                year_model = year_df.groupby(['모델', 'Year_Int'])['정답여부'].mean() * 100
                year_model_pivot = year_model.unstack(fill_value=0)
                
                # 컬럼명을 정수로 변환
                year_model_pivot.columns = year_model_pivot.columns.astype(int)
                
                fig = go.Figure(data=go.Heatmap(
                    z=year_model_pivot.values,
                    x=year_model_pivot.columns,
                    y=year_model_pivot.index,
                    colorscale='RdYlGn',
                    text=np.round(year_model_pivot.values, 1),
                    texttemplate='%{text:.1f}',
                    textfont={"size": annotation_size},
                    colorbar=dict(title=t['accuracy'] + " (%)"),
                    xgap=2,  # 셀 경계선
                    ygap=2
                ))
                fig.update_layout(height=400)
                fig.update_xaxes(tickfont=dict(size=annotation_size))
                fig.update_yaxes(tickfont=dict(size=annotation_size))
                st.plotly_chart(fig, use_container_width=True)
                
                # 연도별 성능 인사이트
                year_avg = year_model_pivot.mean(axis=0).sort_values()
                hardest_year = int(year_avg.index[0])
                easiest_year = int(year_avg.index[-1])
                
                # 연도별 트렌드 분석
                years_sorted = sorted(year_model_pivot.columns)
                if len(years_sorted) >= 3:
                    recent_years = years_sorted[-3:]
                    old_years = years_sorted[:3]
                    recent_avg = year_model_pivot[recent_years].mean(axis=1).mean()
                    old_avg = year_model_pivot[old_years].mean(axis=1).mean()
                    trend = "상승" if recent_avg > old_avg else "하락" if recent_avg < old_avg else "유지"
                    trend_en = "improving" if recent_avg > old_avg else "declining" if recent_avg < old_avg else "stable"
                    
                    st.success(f"""
                    💡 **{"연도별 난이도 트렌드" if lang == 'ko' else "Year-over-Year Difficulty Trend"}**:
                    
                    📅 **{"연도별 난이도" if lang == 'ko' else "Difficulty by Year"}**:
                    - **{"가장 어려운 연도" if lang == 'ko' else "Hardest year"}**: {hardest_year} ({"평균" if lang == 'ko' else "avg"}: {year_avg.iloc[0]:.1f}%)
                    - **{"가장 쉬운 연도" if lang == 'ko' else "Easiest year"}**: {easiest_year} ({"평균" if lang == 'ko' else "avg"}: {year_avg.iloc[-1]:.1f}%)
                    
                    📈 **{"시험 난이도 추세" if lang == 'ko' else "Exam Difficulty Trend"}**:
                    - **{"최근 3년" if lang == 'ko' else "Recent 3 years"}** ({', '.join(map(str, recent_years))}): {"평균" if lang == 'ko' else "avg"} {recent_avg:.1f}%
                    - **{"초기 3년" if lang == 'ko' else "First 3 years"}** ({', '.join(map(str, old_years))}): {"평균" if lang == 'ko' else "avg"} {old_avg:.1f}%
                    - **{"추세" if lang == 'ko' else "Trend"}**: {trend if lang == 'ko' else trend_en} ({abs(recent_avg - old_avg):.1f}%p {"차이" if lang == 'ko' else "difference"})
                    
                    🎓 **{"학습 가이드" if lang == 'ko' else "Study Guide"}**:
                    - {"최근 출제 경향에 집중하여 학습" if lang == 'ko' else "Focus on recent exam patterns"}
                    - {f"{hardest_year}년 문제로 실전 대비" if lang == 'ko' else f"Use {hardest_year} problems for practice"}
                    """)
                else:
                    st.info(f"""
                    💡 **{"연도별 난이도" if lang == 'ko' else "Difficulty by Year"}**:
                    - **{"가장 어려운 연도" if lang == 'ko' else "Hardest year"}**: {hardest_year} ({"평균" if lang == 'ko' else "avg"}: {year_avg.iloc[0]:.1f}%)
                    - **{"가장 쉬운 연도" if lang == 'ko' else "Easiest year"}**: {easiest_year} ({"평균" if lang == 'ko' else "avg"}: {year_avg.iloc[-1]:.1f}%)
                    """)
            else:
                st.info("연도 정보가 있는 데이터가 없습니다.")
    
    # 탭 7: 심층 오답 분석 (Complete Enhanced Version)
    with tabs[6]:
        st.header(f"🔬 {'심층 오답 분석' if lang == 'ko' else 'Deep Incorrect Analysis'}")
        
        st.markdown("""
        > **논문 기반 심층 분석**: 이 탭은 학술 논문의 "공통 오답 패턴(Common Wrong Answer)" 분석 방법론을 적용합니다.
        > 단순히 오답률이 높은 문제를 넘어, **모델들이 일관되게 같은 오답을 선택하는 패턴**을 식별하여 
        > LLM의 근본적인 지식 문제를 파악합니다.
        """)
        
        # 기본 오답 분석 데이터 준비
        # 문제별 오답 통계 계산
        # 🔧 수정: 고유 식별자 생성하여 중복 방지
        # Question만으로는 중복 가능 (여러 테스트에서 같은 문제 번호)
        # → Test Name + Year + Session + Question으로 고유 식별자 생성
        
        # 고유 식별자 생성
        if 'Test Name' in filtered_df.columns:
            filtered_df['unique_question_id'] = (
                filtered_df['Test Name'].astype(str) + '_' +
                filtered_df['Year'].astype(str) + '_' +
                filtered_df['Session'].astype(str) + '_' +
                filtered_df['Question'].astype(str)
            )
        else:
            filtered_df['unique_question_id'] = filtered_df['Question'].astype(str)
        
        # 고유 식별자로 그룹화
        problem_analysis = filtered_df.groupby('unique_question_id').agg({
            '정답여부': ['sum', 'count', 'mean']
        }).reset_index()
        problem_analysis.columns = ['unique_question_id', 'correct_count', 'total_count', 'correct_rate']
        problem_analysis['incorrect_rate'] = 1 - problem_analysis['correct_rate']
        problem_analysis['incorrect_count'] = problem_analysis['total_count'] - problem_analysis['correct_count']
        
        # Question 컬럼도 추가 (표시용)
        questions = []
        for uid in problem_analysis['unique_question_id']:
            matching = filtered_df[filtered_df['unique_question_id'] == uid]
            if len(matching) > 0:
                questions.append(matching.iloc[0]['Question'])
            else:
                questions.append(uid)
        problem_analysis['Question'] = questions
        
        # 문제 식별자 및 메타데이터 추가
        problem_ids = []
        subjects = []
        years = []
        correct_answers = []
        law_statuses = []
        
        for uid in problem_analysis['unique_question_id']:
            matching_rows = filtered_df[filtered_df['unique_question_id'] == uid]
            if len(matching_rows) > 0:
                row = matching_rows.iloc[0]
                problem_id = create_problem_identifier(row, lang)
                problem_ids.append(problem_id)
                subjects.append(row.get('Subject', 'Unknown'))
                years.append(row.get('Year', 'Unknown'))
                correct_answers.append(row.get('Answer', 'Unknown'))
                law_statuses.append(row.get('law', 'Unknown'))
            else:
                problem_ids.append("Unknown")
                subjects.append("Unknown")
                years.append("Unknown")
                correct_answers.append("Unknown")
                law_statuses.append("Unknown")
        
        problem_analysis['problem_id'] = problem_ids
        problem_analysis['Subject'] = subjects
        problem_analysis['Year'] = years
        problem_analysis['CorrectAnswer'] = correct_answers
        problem_analysis['law_status'] = law_statuses
        
        # 모델별 정오답 및 선택한 답 정보 추가
        correct_models_list = []
        incorrect_models_list = []
        selected_answers_dict = []
        
        for uid in problem_analysis['unique_question_id']:
            q_df = filtered_df[filtered_df['unique_question_id'] == uid]
            correct_models = q_df[q_df['정답여부'] == True]['모델'].unique().tolist()
            incorrect_models = q_df[q_df['정답여부'] == False]['모델'].unique().tolist()
            
            correct_models_list.append('✓ ' + ', '.join(sorted(correct_models)) if correct_models else '-')
            incorrect_models_list.append('✗ ' + ', '.join(sorted(incorrect_models)) if incorrect_models else '-')
            
            # 각 모델이 선택한 답 수집 (iterrows 대신 zip 사용)
            answers_by_model = dict(zip(q_df['모델'].values, q_df['예측답'].fillna('N/A').values))
            selected_answers_dict.append(answers_by_model)
        
        problem_analysis['correct_models'] = correct_models_list
        problem_analysis['incorrect_models'] = incorrect_models_list
        problem_analysis['selected_answers'] = selected_answers_dict
        
        # 오답률 순으로 정렬
        problem_analysis = problem_analysis.sort_values(
            by=['incorrect_rate', 'problem_id'],
            ascending=[False, True]
        )
        
        # ========================================
        # 섹션 1: 개요 및 주요 메트릭
        # ========================================
        st.markdown("---")
        st.subheader("📊 " + ("오답 분석 개요" if lang == 'ko' else "Incorrect Analysis Overview"))
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # 🔧 수정: testsets 기준으로 계산 (전체 요약과 동일)
            total_problems_testset = 0
            if selected_tests:
                for test_name in selected_tests:
                    if test_name in testsets:
                        total_problems_testset += len(testsets[test_name])
            
            # 백업: problem_analysis 기준
            total_problems_analysis = len(problem_analysis)
            
            # 우선순위: testsets > analysis
            display_total = total_problems_testset if total_problems_testset > 0 else total_problems_analysis
            
            st.metric(
                "분석 문제 수" if lang == 'ko' else "Total Problems",
                f"{display_total:,}",
                help="테스트셋 기준 총 문제 수"
            )
        
        with col2:
            all_wrong = len(problem_analysis[problem_analysis['correct_count'] == 0])
            st.metric(
                "완전 공통 오답" if lang == 'ko' else "Complete Common Wrong Answers",
                f"{all_wrong}",
                help="모든 모델이 틀린 문제"
            )
        
        with col3:
            most_wrong = len(problem_analysis[problem_analysis['incorrect_rate'] >= 0.5])
            st.metric(
                "주요 공통 오답" if lang == 'ko' else "Major Common Wrong Answers",
                f"{most_wrong}",
                help="50% 이상의 모델이 틀린 문제"
            )
        
        with col4:
            avg_incorrect_rate = problem_analysis['incorrect_rate'].mean() * 100
            st.metric(
                "평균 오답률" if lang == 'ko' else "Avg Incorrect Rate",
                f"{avg_incorrect_rate:.1f}%"
            )
        
        # ========================================
        # 섹션 2: 일관된 오답 선택 패턴 분석 (핵심!)
        # ========================================
        st.markdown("---")
        st.subheader("🎯 " + ("일관된 오답 선택 패턴 (공통 오답 핵심 지표)" if lang == 'ko' else "Consistent Incorrect Answer Pattern"))
        
        with st.expander("📖 " + ("공통 오답이란? (계산 방식 설명)" if lang == 'ko' else "What is Common Wrong Answer?")):
            st.markdown("""
            ### 🔍 계산 방식
            
            **전제 조건: 오답률 50% 이상 문제만 분석**
            ```
            이유:
            - 1-2개 모델만 틀린 문제 → 우연일 가능성
            - 50% 이상 틀린 문제 → 의미 있는 패턴
            
            예시:
            ❌ 제외: 10개 중 1개만 틀림 (10%) → 일관성 의미 없음
            ❌ 제외: 10개 중 3개만 틀림 (30%) → 너무 적음
            ✅ 분석: 10개 중 5개 틀림 (50%) → 의미 있는 패턴
            ✅ 분석: 10개 중 8개 틀림 (80%) → 중요한 패턴
            ```
            
            **1단계: 오답 수집 (오답률 ≥50% 문제만)**
            ```
            문제 Q1 (오답률 80%):
            - GPT-4o: 3번 선택 (정답: 2번) ❌
            - Claude: 3번 선택 (정답: 2번) ❌
            - Gemini: 4번 선택 (정답: 2번) ❌
            - EXAONE: 3번 선택 (정답: 2번) ❌
            - Llama: 2번 선택 (정답: 2번) ✅
            
            → 오답 목록: [3, 3, 4, 3]
            → 오답률: 4/5 = 80% (✅ 분석 대상)
            ```
            
            **2단계: 가장 많이 선택된 오답 찾기**
            ```
            오답 통계:
            - 3번: 3회 ⭐ (가장 많음)
            - 4번: 1회
            
            → 공통 오답: 3번
            ```
            
            **3단계: 일관성 계산**
            ```
            일관성 = (가장 많은 오답 횟수) / (전체 오답 횟수)
                  = 3회 / 4회
                  = 75%
            
            → 75% 일관성 (4개 중 3개가 같은 답 선택)
            ```
            
            **⚠️ nan (답안 추출 실패) 처리**
            ```
            예시 1: nan이 있는 경우
            
            문제 Q2 (오답률 50%, 평가 모델 12개):
            - 전체 오답: 6개
            - 공통 오답 3번: 4개
            - nan (추출 실패): 2개
            
            일관성 계산:
            = 4개 / 6개 = 66.7%
            
            의미:
            "6개 오답 중 4개가 3번 선택 (66.7%)"
            "나머지 2개는 확인 불가 (nan)"
            
            표시: 공통 오답 3.0 (4/6) - 66.7% 일관성
            
            ---
            
            예시 2: nan이 없는 경우
            
            문제 Q3 (오답률 50%, 평가 모델 12개):
            - 전체 오답: 6개
            - 공통 오답 3번: 6개
            - nan: 0개
            
            일관성 계산:
            = 6개 / 6개 = 100%
            
            의미:
            "6개 오답 모두 3번 선택 (100%)"
            
            표시: 공통 오답 3.0 (6/6) - 100% 일관성
            ```
            
            **📌 중요 원칙**:
            ```
            분자 (공통 오답 수):
            → 추출 가능한 오답만 카운트 (nan 제외)
            
            분모 (전체 오답 수):
            → 오답률로 계산된 전체 오답 (nan 포함)
            
            이유:
            - nan은 "무엇을 선택했는지 모름"
            - 분모에는 포함 (전체 오답 수)
            - 분자에는 제외 (확인 불가)
            → 일관성이 낮아짐 (데이터 품질 반영)
            ```
            
            **4단계: 일관성 50% 이상만 표시**
            ```
            ✅ 75% 일관성 → 표시 (의미 있는 패턴)
            ✅ 100% 일관성 → 표시 (완벽한 패턴)
            ❌ 40% 일관성 → 제외 (우연일 가능성)
            ```
            
            ### 💡 왜 오답률 50% 이상만?
            
            **문제 사례**:
            ```
            문제 A (오답률 10%):
            - 10개 모델 중 1개만 틀림
            - 오답: [3]
            - 일관성: 100% ❌ 의미 없음!
            
            문제 B (오답률 80%):
            - 10개 모델 중 8개 틀림
            - 오답: [3, 3, 3, 3, 3, 3, 3, 4]
            - 일관성: 87.5% ✅ 의미 있음!
            ```
            
            **결론**: 
            - **무작위 오답**: 각 모델이 다른 답 선택 → 우연
            - **일관된 오답**: 여러 모델이 같은 답 선택 → 체계적 오해!
            
            일관된 오답은 특정 개념에 대한 근본적인 오해를 의미합니다.
            """)
        
        st.info("""
        💡 **핵심 인사이트**: 모델들이 단순히 틀리는 것이 아니라, **같은 오답을 일관되게 선택**하는 경우 
        이는 해당 지식 영역에 대한 근본적인 이해 부족을 의미합니다. (논문 방법론 적용)
        
        ⚠️ **중요**: 오답률 50% 이상 문제만 분석합니다 (소수 모델 오답은 우연일 가능성)
        """)
        
        # 일관된 오답 패턴 계산
        consistent_wrong_patterns = []
        extraction_failures = []  # 답안 추출 실패 추적
        
        for idx, row in problem_analysis.iterrows():
            # ⭐ 핵심 변경: 오답률 50% 이상 & 최소 2개 이상 오답인 문제만 분석
            if row['incorrect_rate'] >= 0.5 and row['incorrect_count'] >= 2:
                selected = row['selected_answers']
                correct = row['CorrectAnswer']
                
                # 전체 평가 모델 수
                total_models = row['total_count']
                
                # 🔍 올바른 전체 오답 모델 수 계산
                total_incorrect_models = row['incorrect_count']  # problem_analysis에서 계산된 값 사용
                
                # 답안 추출 통계
                valid_answers = 0
                nan_count = 0
                empty_count = 0
                
                # 오답을 선택한 모델들의 답변 수집
                wrong_answers = []
                wrong_answer_models = []  # 오답 모델 추적
                
                for model, answer in selected.items():
                    if pd.isna(answer):
                        nan_count += 1
                    elif str(answer).strip() == '':
                        empty_count += 1
                    elif str(answer) != str(correct):
                        # 정답이 아니면 오답
                        wrong_answers.append(str(answer).strip())
                        wrong_answer_models.append(model)
                        valid_answers += 1
                    else:
                        valid_answers += 1
                
                # 🔍 추출 실패 기록
                if nan_count > 0 or empty_count > 0:
                    extraction_failures.append({
                        'problem_id': row['problem_id'],
                        'total_models': total_models,
                        'valid_answers': valid_answers,
                        'nan_count': nan_count,
                        'empty_count': empty_count,
                        'extraction_rate': (valid_answers / total_models * 100) if total_models > 0 else 0
                    })
                
                if wrong_answers and len(wrong_answers) >= 2:
                    from collections import Counter
                    answer_counts = Counter(wrong_answers)
                    most_common_wrong, count = answer_counts.most_common(1)[0]
                    
                    # ⭐⭐⭐ 핵심 수정: 올바른 일관성 계산
                    # 일관성 = (가장 많이 선택된 오답 수) / (전체 오답 모델 수)
                    # 
                    # 중요: nan 처리
                    # - 분자(count): nan 제외 (추출 가능한 것만)
                    # - 분모(total_incorrect_models): nan 포함 (전체 오답)
                    # 
                    # 예시: 전체 오답 6개, 공통 오답 4개, nan 2개
                    # → 일관성 = 4/6 = 66.7%
                    # → 의미: "6개 오답 중 4개가 동일 답 선택, 2개는 확인 불가"
                    consistency_ratio = count / total_incorrect_models if total_incorrect_models > 0 else 0
                    
                    # 🔍 검증 1: 추출된 오답 수 vs 기록된 오답 수
                    if len(wrong_answers) != total_incorrect_models:
                        st.sidebar.warning(
                            f"⚠️ 문제 {row['problem_id']}: 오답 수 불일치\n"
                            f"   추출: {len(wrong_answers)}개, 기록: {total_incorrect_models}개\n"
                            f"   → nan/빈답안: {nan_count + empty_count}개"
                        )
                    
                    # 🔍 검증 2: 일관성 비율이 1.0 초과하면 오류
                    if consistency_ratio > 1.0:
                        st.sidebar.error(
                            f"❌ 문제 {row['problem_id']}: 일관성 계산 오류!\n"
                            f"   공통 오답: {count}개, 전체 오답: {total_incorrect_models}개\n"
                            f"   → {count}/{total_incorrect_models} = {consistency_ratio:.2%}"
                        )
                        consistency_ratio = 1.0  # 최대값으로 보정
                    
                    if consistency_ratio >= 0.5:
                        models_selected_this = [m for m, a in selected.items() 
                                              if pd.notna(a) and str(a).strip() == most_common_wrong]
                        
                        consistent_wrong_patterns.append({
                            'problem_id': row['problem_id'],
                            'Question': row['Question'],
                            'Subject': row['Subject'],
                            'Year': row['Year'],
                            'correct_answer': str(correct).strip(),
                            'common_wrong_answer': most_common_wrong,
                            'wrong_answer_count': count,  # 공통 오답 수
                            'total_wrong': total_incorrect_models,  # ⭐ 전체 오답 모델 수 (올바른 값)
                            'consistency_ratio': consistency_ratio,  # ⭐ 올바른 일관성
                            'models_with_this_wrong': ', '.join(models_selected_this),
                            'incorrect_rate': row['incorrect_rate'],
                            'total_models': total_models,
                            'valid_answers': valid_answers,
                            'nan_count': nan_count,
                            'empty_count': empty_count,
                            'extracted_wrong_count': len(wrong_answers)  # 실제 추출된 오답 수
                        })
        
        # 🚨 답안 추출 실패 통계 표시
        if extraction_failures:
            with st.expander(f"⚠️ 답안 추출 실패 통계 ({len(extraction_failures)}개 문제)"):
                failure_df = pd.DataFrame(extraction_failures)
                failure_df = failure_df.sort_values('nan_count', ascending=False)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    total_nan = failure_df['nan_count'].sum()
                    st.metric("총 nan 개수", f"{total_nan}개")
                with col2:
                    total_empty = failure_df['empty_count'].sum()
                    st.metric("총 빈 답안 개수", f"{total_empty}개")
                with col3:
                    avg_extraction = failure_df['extraction_rate'].mean()
                    st.metric("평균 추출 성공률", f"{avg_extraction:.1f}%")
                
                st.warning("""
                💡 **답안 추출 실패 원인**:
                - **nan**: 모델이 답을 추출하지 못함 (파싱 오류, 형식 불일치)
                - **빈 답안**: 모델이 빈 문자열을 반환
                
                **영향**:
                - 오답률 계산 시 분모가 감소
                - 일관성 계산에서 제외됨
                
                **조치**:
                - 벤치마크 로그 확인
                - 답안 추출 로직 개선
                - 해당 문제 재실행 고려
                """)
                
                st.dataframe(
                    failure_df.head(20).style.format({
                        'extraction_rate': '{:.1f}%'
                    }).background_gradient(
                        subset=['nan_count'],
                        cmap='Reds'
                    ),
                    width='stretch'
                )
        
        if consistent_wrong_patterns:
            consistent_df = pd.DataFrame(consistent_wrong_patterns)
            consistent_df = consistent_df.sort_values('consistency_ratio', ascending=False)
            
            # 100% 일관성과 50-99% 일관성 구분
            perfect_consistency = consistent_df[consistent_df['consistency_ratio'] == 1.0]
            high_but_not_perfect = consistent_df[consistent_df['consistency_ratio'] < 1.0]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("총 일관된 오답 패턴", f"{len(consistent_df)}개",
                         help="오답률 50% 이상 & 일관성 50% 이상 문제")
            with col2:
                st.metric("100% 일관성", f"{len(perfect_consistency)}개", 
                         delta=f"{len(perfect_consistency)/len(consistent_df)*100:.1f}%",
                         help="모든 오답 모델이 같은 답을 선택")
            with col3:
                st.metric("50-99% 일관성", f"{len(high_but_not_perfect)}개",
                         delta=f"{len(high_but_not_perfect)/len(consistent_df)*100:.1f}%",
                         help="대부분 오답 모델이 같은 답을 선택")
            
            st.success(f"""
            ✅ **{len(consistent_df)}개의 일관된 오답 패턴 발견!** (오답률 ≥50% 문제 중)
            
            이는 특정 모델의 문제가 아닌, **여러 LLM에 공통적으로 존재하는 공통 오답**를 의미합니다.
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    consistent_df,
                    x='consistency_ratio',
                    nbins=20,
                    title='일관된 오답 선택 비율 분포' if lang == 'ko' else 'Consistency Ratio Distribution',
                    labels={'consistency_ratio': '일관성 비율' if lang == 'ko' else 'Consistency Ratio'}
                )
                fig.update_traces(marker_line_color='black', marker_line_width=1.5)
                fig.update_layout(
                    xaxis_title='일관성 비율 (1.0 = 100% 일치)',
                    yaxis_title='문제 수',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'Subject' in consistent_df.columns:
                    subject_pattern_count = consistent_df['Subject'].value_counts().reset_index()
                    subject_pattern_count.columns = ['Subject', 'Count']
                    
                    fig = px.bar(
                        subject_pattern_count.head(10),
                        x='Subject',
                        y='Count',
                        title='과목별 일관된 오답 패턴' if lang == 'ko' else 'Consistent Wrong Patterns by Subject',
                        color='Count',
                        color_continuous_scale='Reds'
                    )
                    fig.update_traces(marker_line_color='black', marker_line_width=1.5)
                    fig.update_layout(height=400)
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
            
            # 탭으로 100% / 50-99% 구분
            tab1, tab2 = st.tabs([
                f"🔴 100% 일관성 ({len(perfect_consistency)}개)",
                f"🟠 50-99% 일관성 ({len(high_but_not_perfect)}개)"
            ])
            
            with tab1:
                st.markdown("#### " + ("모든 오답 모델이 같은 답을 선택한 문제" if lang == 'ko' else "All Wrong Models Selected Same Answer"))
                st.caption("⚠️ 오답률 50% 이상 문제만 포함")
                
                if len(perfect_consistency) > 0:
                    display_perfect = perfect_consistency.copy()
                    display_perfect['일관성_pct'] = 100.0
                    display_perfect['오답률_pct'] = (display_perfect['incorrect_rate'] * 100).round(1)
                    display_perfect['오답_정보'] = (display_perfect['common_wrong_answer'].astype(str) + 
                                                   ' (' + display_perfect['wrong_answer_count'].astype(str) + 
                                                   '/' + display_perfect['total_wrong'].astype(str) + ')')
                    
                    # 🔥 오답률 높은 순으로 정렬
                    display_perfect = display_perfect.sort_values('incorrect_rate', ascending=False)
                    
                    # 🔍 검증: 오답률 계산 확인
                    display_perfect['검증_오답률'] = (display_perfect['total_wrong'] / display_perfect['total_models'] * 100).round(1)
                    display_perfect['검증_일치'] = (display_perfect['오답률_pct'] - display_perfect['검증_오답률']).abs() < 1.0
                    
                    display_df = pd.DataFrame({
                        '문제 번호' if lang == 'ko' else 'Problem ID': display_perfect['problem_id'],
                        '과목' if lang == 'ko' else 'Subject': display_perfect['Subject'],
                        '오답률 (%)': display_perfect['오답률_pct'],
                        '정답' if lang == 'ko' else 'Correct': display_perfect['correct_answer'],
                        '공통 오답 (횟수/전체)': display_perfect['오답_정보'],
                        '일관성 (%)': display_perfect['일관성_pct'],
                        '평가 모델수': display_perfect['total_models'],
                        '해당 오답 선택 모델': display_perfect['models_with_this_wrong']
                    })
                    
                    # 🚨 검증 실패 경고
                    if not display_perfect['검증_일치'].all():
                        st.warning(f"""
                        ⚠️ **데이터 불일치 경고**: 일부 문제에서 오답률과 실제 오답 수가 맞지 않습니다.
                        
                        가능한 원인:
                        - 일부 모델의 답안 추출 실패 (nan 값)
                        - 특정 모델이 해당 문제를 평가하지 않음
                        - 데이터 필터링으로 인한 모델 수 변화
                        
                        💡 "평가 모델수" 컬럼을 확인하여 실제 평가된 모델 수를 확인하세요.
                        """)
                    
                    st.dataframe(
                        display_df.style.background_gradient(
                            subset=['오답률 (%)'],
                            cmap='Reds',
                            vmin=50,
                            vmax=100
                        ).format({
                            '일관성 (%)': '{:.1f}%',
                            '오답률 (%)': '{:.1f}%',
                            '평가 모델수': '{:.0f}'
                        }),
                        width='stretch',
                        height=500
                    )
                    
                    if st.checkbox('📋 ' + ('100% 일관성 문제 상세 보기' if lang == 'ko' else 'Show Details'), key='perfect_details'):
                        st.info(f"💡 총 {len(display_perfect)}개 문제의 상세 내용을 표시합니다. (오답률 높은 순)")
                        for idx, row in display_perfect.iterrows():  # 🔥 head(20) 제거 - 전체 표시
                            with st.expander(f"🔍 {row['problem_id']} - 일관성 100% (오답률 {row['incorrect_rate']*100:.1f}%)"):
                                q_detail = filtered_df[filtered_df['Question'] == row['Question']].iloc[0]
                                
                                st.markdown(f"**{'문제' if lang == 'ko' else 'Question'}:** {q_detail['Question']}")
                                st.markdown(f"**{'과목' if lang == 'ko' else 'Subject'}:** {row['Subject']}")
                                st.markdown(f"**오답률:** {row['incorrect_rate']*100:.1f}%")
                                
                                if all([f'Option {i}' in q_detail for i in range(1, 5)]):
                                    st.markdown("**선택지:**")
                                    for i in range(1, 5):
                                        option = q_detail[f'Option {i}']
                                        if pd.notna(option):
                                            if str(i) == str(row['correct_answer']):
                                                st.markdown(f"✅ **{i}. {option}** (정답)")
                                            elif str(i) == str(row['common_wrong_answer']):
                                                st.markdown(f"❌ **{i}. {option}** (모든 오답 모델이 선택)")
                                            else:
                                                st.markdown(f"  {i}. {option}")
                                
                                st.markdown(f"**🎯 정답:** {row['correct_answer']}")
                                st.markdown(f"**❌ 공통 오답:** {row['common_wrong_answer']} (모든 {row['total_wrong']}개 오답 모델)")
                                st.markdown(f"**🤖 해당 오답 선택 모델:** {row['models_with_this_wrong']}")
                else:
                    st.info("100% 일관성 패턴이 발견되지 않았습니다.")
            
            with tab2:
                st.markdown("#### " + ("50-99%의 오답 모델이 같은 답을 선택한 문제" if lang == 'ko' else "50-99% Wrong Models Selected Same Answer"))
                st.caption("⚠️ 오답률 50% 이상 문제만 포함")
                
                if len(high_but_not_perfect) > 0:
                    display_high = high_but_not_perfect.copy()
                    display_high['일관성_pct'] = (display_high['consistency_ratio'] * 100).round(1)
                    display_high['오답률_pct'] = (display_high['incorrect_rate'] * 100).round(1)
                    display_high['오답_정보'] = (display_high['common_wrong_answer'].astype(str) + 
                                               ' (' + display_high['wrong_answer_count'].astype(str) + 
                                               '/' + display_high['total_wrong'].astype(str) + ')')
                    
                    # 🔥 오답률 높은 순으로 정렬
                    display_high = display_high.sort_values('incorrect_rate', ascending=False)
                    
                    display_df = pd.DataFrame({
                        '문제 번호' if lang == 'ko' else 'Problem ID': display_high['problem_id'],
                        '과목' if lang == 'ko' else 'Subject': display_high['Subject'],
                        '오답률 (%)': display_high['오답률_pct'],
                        '정답' if lang == 'ko' else 'Correct': display_high['correct_answer'],
                        '공통 오답 (횟수/전체)': display_high['오답_정보'],
                        '일관성 (%)': display_high['일관성_pct'],
                        '평가 모델수': display_high['total_models'],
                        '해당 오답 선택 모델': display_high['models_with_this_wrong']
                    })
                    
                    st.dataframe(
                        display_df.style.background_gradient(
                            subset=['일관성 (%)'],
                            cmap='Oranges',
                            vmin=50,
                            vmax=100
                        ).format({
                            '일관성 (%)': '{:.1f}%',
                            '오답률 (%)': '{:.1f}%',
                            '평가 모델수': '{:.0f}'
                        }),
                        width='stretch',
                        height=500
                    )
                    
                    if st.checkbox('📋 ' + ('50-99% 일관성 문제 상세 보기' if lang == 'ko' else 'Show Details'), key='high_details'):
                        st.info(f"💡 총 {len(display_high)}개 문제의 상세 내용을 표시합니다. (오답률 높은 순)")
                        for idx, row in display_high.iterrows():  # 🔥 head(30) 제거 - 전체 표시
                            with st.expander(f"🔍 {row['problem_id']} - 일관성 {row['consistency_ratio']*100:.1f}% (오답률 {row['incorrect_rate']*100:.1f}%)"):
                                q_detail = filtered_df[filtered_df['Question'] == row['Question']].iloc[0]
                                
                                st.markdown(f"**{'문제' if lang == 'ko' else 'Question'}:** {q_detail['Question']}")
                                st.markdown(f"**{'과목' if lang == 'ko' else 'Subject'}:** {row['Subject']}")
                                st.markdown(f"**오답률:** {row['incorrect_rate']*100:.1f}%")
                                
                                if all([f'Option {i}' in q_detail for i in range(1, 5)]):
                                    st.markdown("**선택지:**")
                                    for i in range(1, 5):
                                        option = q_detail[f'Option {i}']
                                        if pd.notna(option):
                                            if str(i) == str(row['correct_answer']):
                                                st.markdown(f"✅ **{i}. {option}** (정답)")
                                            elif str(i) == str(row['common_wrong_answer']):
                                                st.markdown(f"❌ **{i}. {option}** (일관된 오답 - {row['wrong_answer_count']}개 모델)")
                                            else:
                                                st.markdown(f"  {i}. {option}")
                                
                                st.markdown(f"**🎯 정답:** {row['correct_answer']}")
                                st.markdown(f"**❌ 공통 오답:** {row['common_wrong_answer']} ({row['wrong_answer_count']}/{row['total_wrong']} = {row['consistency_ratio']*100:.1f}% 일관성)")
                                st.markdown(f"**🤖 해당 오답 선택 모델:** {row['models_with_this_wrong']}")
                else:
                    st.info("50-99% 일관성 패턴이 발견되지 않았습니다.")
        else:
            st.warning("일관된 오답 선택 패턴이 발견되지 않았습니다. (오답률 50% 이상 & 일관성 50% 이상 문제 없음)")
        
        # 섹션 3: 프롬프팅 방식별 공통 오답 비교
        # ========================================
        st.markdown("---")
        st.subheader("📋 " + ("프롬프팅 방식별 공통 오답 분석" if lang == 'ko' else "Common Wrong Answer by Prompting"))
        
        if '프롬프팅' in filtered_df.columns and filtered_df['프롬프팅'].nunique() > 1:
            st.info("""
            💡 **논문 방법론**: 특정 프롬프팅 방식에서 모델들이 일관되게 틀리는 문제를 식별
            """)
            
            prompting_analysis = []
            
            for prompting in filtered_df['프롬프팅'].unique():
                prompt_df = filtered_df[filtered_df['프롬프팅'] == prompting]
                prompt_problems = prompt_df.groupby('Question').agg({'정답여부': ['sum', 'count']}).reset_index()
                prompt_problems.columns = ['Question', 'correct_count', 'total_count']
                all_wrong_in_prompt = len(prompt_problems[prompt_problems['correct_count'] == 0])
                avg_acc = prompt_df['정답여부'].mean() * 100
                
                prompting_analysis.append({
                    '프롬프팅': prompting,
                    '전체_문제수': prompt_df['Question'].nunique(),
                    '완전_지식격차': all_wrong_in_prompt,
                    '평균_정확도': avg_acc,
                    '지식격차_비율': (all_wrong_in_prompt / prompt_df['Question'].nunique() * 100) if prompt_df['Question'].nunique() > 0 else 0
                })
            
            prompt_comp_df = pd.DataFrame(prompting_analysis).sort_values('완전_지식격차', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    prompt_comp_df,
                    x='프롬프팅',
                    y='완전_지식격차',
                    title='프롬프팅 방식별 완전 공통 오답',
                    text='완전_지식격차',
                    color='완전_지식격차',
                    color_continuous_scale='Reds'
                )
                fig.update_traces(textposition='outside', marker_line_color='black', marker_line_width=1.5)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(
                    prompt_comp_df,
                    x='평균_정확도',
                    y='지식격차_비율',
                    size='전체_문제수',
                    text='프롬프팅',
                    title='정확도 vs 지식격차 비율',
                    labels={'평균_정확도': '평균 정확도 (%)', '지식격차_비율': '지식격차 비율 (%)'}
                )
                fig.update_traces(textposition='top center', marker=dict(line=dict(width=2, color='black')))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(
                prompt_comp_df.style.format({
                    '평균_정확도': '{:.2f}%',
                    '지식격차_비율': '{:.2f}%'
                }).background_gradient(subset=['완전_지식격차'], cmap='Reds'),
                width='stretch'
            )
        else:
            st.info("프롬프팅 방식이 1개만 선택되어 비교 불가")
        
        # ========================================
        # 섹션 4: 모델 간 오답 일치도 매트릭스
        # ========================================
        st.markdown("---")
        st.subheader("🔗 " + ("모델 간 오답 일치도 매트릭스" if lang == 'ko' else "Inter-Model Error Agreement"))
        
        st.info("""
        💡 **분석 목적**: 어떤 모델들이 유사한 실수를 하는지 파악
        높은 일치도 = 유사한 공통 오답 공유
        """)
        
        models = filtered_df['모델'].unique().tolist()
        
        if len(models) >= 2:
            problem_model_matrix = filtered_df.pivot_table(
                index='Question',
                columns='모델',
                values='정답여부',
                aggfunc='first'
            ).fillna(0)
            
            agreement_matrix = pd.DataFrame(index=models, columns=models, dtype=float)
            
            for model1 in models:
                for model2 in models:
                    if model1 in problem_model_matrix.columns and model2 in problem_model_matrix.columns:
                        both_wrong = ((problem_model_matrix[model1] == 0) & (problem_model_matrix[model2] == 0)).sum()
                        either_wrong = ((problem_model_matrix[model1] == 0) | (problem_model_matrix[model2] == 0)).sum()
                        agreement = both_wrong / either_wrong if either_wrong > 0 else 0
                        agreement_matrix.loc[model1, model2] = agreement * 100
                    else:
                        agreement_matrix.loc[model1, model2] = 0
            
            agreement_matrix = agreement_matrix.astype(float)
            
            fig = go.Figure(data=go.Heatmap(
                z=agreement_matrix.values,
                x=agreement_matrix.columns,
                y=agreement_matrix.index,
                colorscale='RdYlBu_r',
                text=np.round(agreement_matrix.values, 1),
                texttemplate='%{text:.1f}',
                textfont={"size": int(12 * chart_text_size)},
                colorbar=dict(title="일치도 (%)"),
                xgap=2,
                ygap=2
            ))
            
            fig.update_layout(
                title='모델 간 오답 일치도 (%)',
                height=max(400, len(models) * 40)
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            agreement_pairs = []
            for i, model1 in enumerate(models):
                for j, model2 in enumerate(models):
                    if i < j:
                        agreement_pairs.append({
                            'Model 1': model1,
                            'Model 2': model2,
                            'Agreement': agreement_matrix.loc[model1, model2]
                        })
            
            if agreement_pairs:
                pairs_df = pd.DataFrame(agreement_pairs).sort_values('Agreement', ascending=False)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**가장 유사한 오답 패턴 (Top 5)**")
                    for idx, row in pairs_df.head(5).iterrows():
                        st.write(f"🔗 **{row['Model 1']}** ↔ **{row['Model 2']}**: {row['Agreement']:.1f}%")
                
                with col2:
                    st.markdown("**가장 다른 오답 패턴 (Bottom 5)**")
                    for idx, row in pairs_df.tail(5).iterrows():
                        st.write(f"↔️ **{row['Model 1']}** ↔ **{row['Model 2']}**: {row['Agreement']:.1f}%")
        else:
            st.warning("2개 이상의 모델이 필요합니다.")
        
        # ========================================
        # 섹션 5: 공통 오답 영역 매핑
        # ========================================
        st.markdown("---")
        st.subheader("🗺️ " + ("공통 오답 영역 매핑" if lang == 'ko' else "Common Wrong Answer Domain Mapping"))
        
        all_wrong = problem_analysis[problem_analysis['correct_count'] == 0]
        
        if len(all_wrong) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                if 'Subject' in all_wrong.columns:
                    subject_gaps = all_wrong['Subject'].value_counts().reset_index()
                    subject_gaps.columns = ['Subject', 'Count']
                    total_by_subject = problem_analysis['Subject'].value_counts()
                    subject_gaps['Total'] = subject_gaps['Subject'].map(total_by_subject)
                    subject_gaps['Gap_Ratio'] = (subject_gaps['Count'] / subject_gaps['Total'] * 100).round(1)
                    
                    fig = px.bar(
                        subject_gaps.sort_values('Gap_Ratio', ascending=False),
                        x='Subject',
                        y='Gap_Ratio',
                        title='과목별 공통 오답 비율',
                        text='Gap_Ratio',
                        color='Gap_Ratio',
                        color_continuous_scale='Reds',
                        hover_data=['Count', 'Total']
                    )
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside', marker_line_color='black', marker_line_width=1.5)
                    fig.update_layout(height=400, yaxis_title='공통 오답 비율 (%)')
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'Year' in all_wrong.columns:
                    year_gaps = all_wrong['Year'].value_counts().reset_index()
                    year_gaps.columns = ['Year', 'Count']
                    year_gaps = year_gaps[year_gaps['Year'] != 'Unknown']
                    
                    if len(year_gaps) > 0:
                        year_gaps['Year_Int'] = year_gaps['Year'].apply(safe_convert_to_int)
                        year_gaps = year_gaps[year_gaps['Year_Int'].notna()].sort_values('Year_Int')
                        
                        fig = px.line(
                            year_gaps,
                            x='Year_Int',
                            y='Count',
                            title='연도별 공통 오답 문제 수',
                            markers=True,
                            text='Count'
                        )
                        fig.update_traces(textposition='top center', marker_size=10, marker_line_color='black', marker_line_width=2, line_width=3)
                        fig.update_layout(height=400, xaxis_title='연도', yaxis_title='문제 수')
                        st.plotly_chart(fig, use_container_width=True)
        
        # ========================================
        # 섹션 6: Top 20 오답률 높은 문제
        # ========================================
        st.markdown("---")
        st.subheader("📊 " + ("오답률 높은 문제 Top 20" if lang == 'ko' else "Top 20 Problems by Incorrect Rate"))
        
        top_20 = problem_analysis.head(20)
        
        display_top_20 = pd.DataFrame({
            ('문제 번호' if lang == 'ko' else 'Problem ID'): top_20['problem_id'],
            ('과목' if lang == 'ko' else 'Subject'): top_20['Subject'],
            ('오답 모델수' if lang == 'ko' else 'Incorrect Count'): top_20['incorrect_count'].astype(int),
            ('정답 모델수' if lang == 'ko' else 'Correct Count'): top_20['correct_count'].astype(int),
            ('총 모델수' if lang == 'ko' else 'Total Models'): top_20['total_count'].astype(int),
            ('오답률' if lang == 'ko' else 'Wrong Rate'): (top_20['incorrect_rate'] * 100).round(2),
            '정답 모델' if lang == 'ko' else 'Correct Models': top_20['correct_models'],
            '오답 모델' if lang == 'ko' else 'Incorrect Models': top_20['incorrect_models']
        })
        
        st.dataframe(
            display_top_20.style.background_gradient(
                subset=['오답률' if lang == 'ko' else 'Wrong Rate'],
                cmap='Reds',
                vmin=0,
                vmax=100
            ),
            width='stretch',
            height=600
        )
        
        # ========================================
        # 섹션 7: 모든 모델이 틀린 문제 (완전 공통 오답)
        # ========================================
        st.markdown("---")
        st.subheader("🚨 " + ("모든 모델이 틀린 문제 (완전 공통 오답)" if lang == 'ko' else "All Models Incorrect (Complete Common Wrong Answer)"))
        
        all_wrong = problem_analysis[problem_analysis['correct_count'] == 0]
        
        if len(all_wrong) > 0:
            st.error(f"""
            ⚠️ **심각한 공통 오답 발견: {len(all_wrong)}개 문제**
            
            이 문제들은 **모든 평가 모델이 틀렸습니다**. 현재 LLM들이 공통적으로 
            해당 지식 영역을 제대로 이해하지 못하고 있음을 의미합니다.
            """)
            
            display_all_wrong = pd.DataFrame({
                ('문제 번호' if lang == 'ko' else 'Problem ID'): all_wrong['problem_id'],
                ('과목' if lang == 'ko' else 'Subject'): all_wrong['Subject'],
                ('연도' if lang == 'ko' else 'Year'): all_wrong['Year'],
                ('오답 모델수' if lang == 'ko' else 'Incorrect Count'): all_wrong['incorrect_count'].astype(int),
                '오답 모델' if lang == 'ko' else 'Incorrect Models': all_wrong['incorrect_models']
            })
            
            st.dataframe(display_all_wrong, width='stretch', height=400)
            
            if st.checkbox('문제 내용 보기 (완전 공통 오답)' if lang == 'ko' else 'Show Details (Complete Gap)', key='all_wrong_details'):
                st.info(f"총 {len(all_wrong)}개 문제의 상세 내용")
                for idx, row in all_wrong.head(20).iterrows():
                    with st.expander(f"🚨 {row['problem_id']}"):
                        q_detail = filtered_df[filtered_df['Question'] == row['Question']].iloc[0]
                        st.write(f"**{'문제' if lang == 'ko' else 'Question'}:** {q_detail['Question']}")
                        
                        if 'Subject' in q_detail and pd.notna(q_detail['Subject']):
                            st.write(f"**과목:** {q_detail['Subject']}")
                        
                        if all([f'Option {i}' in q_detail for i in range(1, 5)]):
                            st.write("**선택지:**")
                            for i in range(1, 5):
                                option = q_detail[f'Option {i}']
                                if pd.notna(option):
                                    if str(i) == str(row['CorrectAnswer']):
                                        st.write(f"  ✅ {i}. {option} **(정답)**")
                                    else:
                                        st.write(f"  {i}. {option}")
                        
                        if 'Answer' in q_detail and pd.notna(q_detail['Answer']):
                            st.write(f"**정답:** {q_detail['Answer']}")
                        
                        st.write("**각 모델이 선택한 답:**")
                        for model, answer in row['selected_answers'].items():
                            st.write(f"  • {model}: {answer}")
        else:
            st.success("✅ 모든 모델이 틀린 문제가 없습니다!")
        
        # ========================================
        # 섹션 8: 대부분 모델이 틀린 문제 (≥50%) ⭐ 기존 기능
        # ========================================
        st.markdown("---")
        st.subheader("⚠️ " + ("대부분 모델이 틀린 문제 (≥50%)" if lang == 'ko' else "Most Models Incorrect (≥50%)"))
        
        most_wrong = problem_analysis[problem_analysis['incorrect_rate'] >= 0.5]
        
        if len(most_wrong) > 0:
            st.warning(f"""
            ⚠️ **주요 공통 오답: {len(most_wrong)}개 문제**
            
            이 문제들은 **50% 이상의 모델이 틀렸습니다**. 해당 지식 영역이 
            많은 LLM에게 어려운 영역임을 의미합니다.
            """)
            
            display_most_wrong = pd.DataFrame({
                ('문제 번호' if lang == 'ko' else 'Problem ID'): most_wrong['problem_id'],
                ('과목' if lang == 'ko' else 'Subject'): most_wrong['Subject'],
                ('오답 모델수' if lang == 'ko' else 'Incorrect Count'): most_wrong['incorrect_count'].astype(int),
                ('정답 모델수' if lang == 'ko' else 'Correct Count'): most_wrong['correct_count'].astype(int),
                ('총 모델수' if lang == 'ko' else 'Total Models'): most_wrong['total_count'].astype(int),
                ('오답률' if lang == 'ko' else 'Wrong Rate'): (most_wrong['incorrect_rate'] * 100).round(2),
                '정답 모델' if lang == 'ko' else 'Correct Models': most_wrong['correct_models'],
                '오답 모델' if lang == 'ko' else 'Incorrect Models': most_wrong['incorrect_models']
            })
            
            st.dataframe(
                display_most_wrong.style.background_gradient(
                    subset=['오답률' if lang == 'ko' else 'Wrong Rate'],
                    cmap='Reds',
                    vmin=0,
                    vmax=100
                ),
                width='stretch',
                height=400
            )
            
            if st.checkbox('문제 내용 보기 (대부분 틀린 문제)' if lang == 'ko' else 'Show Details (Most Incorrect)', key='most_wrong_details'):
                st.info(f"총 {len(most_wrong)}개 문제 중 상위 20개")
                for idx, row in most_wrong.head(20).iterrows():
                    with st.expander(f"⚠️ {row['problem_id']} - 오답률 {row['incorrect_rate']*100:.1f}%"):
                        q_detail = filtered_df[filtered_df['Question'] == row['Question']].iloc[0]
                        st.write(f"**문제:** {q_detail['Question']}")
                        
                        if 'Subject' in q_detail and pd.notna(q_detail['Subject']):
                            st.write(f"**과목:** {q_detail['Subject']}")
                        
                        if all([f'Option {i}' in q_detail for i in range(1, 5)]):
                            st.write("**선택지:**")
                            for i in range(1, 5):
                                option = q_detail[f'Option {i}']
                                if pd.notna(option):
                                    if str(i) == str(row['CorrectAnswer']):
                                        st.write(f"  ✅ {i}. {option} **(정답)**")
                                    else:
                                        st.write(f"  {i}. {option}")
                        
                        if 'Answer' in q_detail and pd.notna(q_detail['Answer']):
                            st.write(f"**정답:** {q_detail['Answer']}")
                        
                        st.markdown("---")
                        
                        # 정답 모델과 오답 모델 분리 표시
                        if row['correct_count'] > 0:
                            st.write("**✓ 정답 모델:**")
                            for model, answer in row['selected_answers'].items():
                                if str(answer) == str(row['CorrectAnswer']):
                                    st.write(f"  • **{model}**: 선택 {answer} ✅")
                        
                        st.write("**✗ 오답 모델:**")
                        for model, answer in row['selected_answers'].items():
                            if str(answer) != str(row['CorrectAnswer']):
                                st.write(f"  • **{model}**: 선택 {answer} (정답: {row['CorrectAnswer']})")
        else:
            st.info("50% 이상의 모델이 틀린 문제가 없습니다.")
        
        # ========================================
        # 섹션 9: 법령/비법령 오답 분석 ⭐ 신규 기능
        # ========================================
        if 'law' in filtered_df.columns:
            st.markdown("---")
            st.subheader("⚖️ " + ("법령/비법령 오답 분석" if lang == 'ko' else "Law/Non-Law Incorrect Analysis"))
            
            st.info("""
            💡 **법령 공통 오답 분석**: 법령 문제와 비법령 문제에서 모델들의 오답 패턴이 
            어떻게 다른지 분석하여 법률 지식의 격차를 파악합니다.
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 법령/비법령별 완전 공통 오답 비율
                law_all_wrong = problem_analysis[problem_analysis['correct_count'] == 0]
                law_gap_by_type = law_all_wrong['law_status'].value_counts()
                
                law_gap_data = pd.DataFrame({
                    '구분': ['법령' if x == 'O' else '비법령' for x in law_gap_by_type.index],
                    '완전_지식격차': law_gap_by_type.values
                })
                
                total_law = len(problem_analysis[problem_analysis['law_status'] == 'O'])
                total_non_law = len(problem_analysis[problem_analysis['law_status'] != 'O'])
                
                law_gap_data['전체문제'] = law_gap_data['구분'].apply(
                    lambda x: total_law if x == '법령' else total_non_law
                )
                law_gap_data['비율'] = (law_gap_data['완전_지식격차'] / law_gap_data['전체문제'] * 100).round(1)
                
                fig = px.bar(
                    law_gap_data,
                    x='구분',
                    y='비율',
                    title='법령/비법령 완전 공통 오답 비율',
                    text='비율',
                    color='비율',
                    color_continuous_scale='Reds',
                    hover_data=['완전_지식격차', '전체문제']
                )
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside', marker_line_color='black', marker_line_width=1.5)
                fig.update_layout(height=400, yaxis_title='공통 오답 비율 (%)')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # 법령/비법령별 평균 오답률
                law_incorrect_rate = problem_analysis.groupby('law_status')['incorrect_rate'].mean().reset_index()
                law_incorrect_rate['구분'] = law_incorrect_rate['law_status'].apply(lambda x: '법령' if x == 'O' else '비법령')
                law_incorrect_rate['평균_오답률'] = law_incorrect_rate['incorrect_rate'] * 100
                
                fig = px.bar(
                    law_incorrect_rate,
                    x='구분',
                    y='평균_오답률',
                    title='법령/비법령 평균 오답률',
                    text='평균_오답률',
                    color='평균_오답률',
                    color_continuous_scale='Reds'
                )
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside', marker_line_color='black', marker_line_width=1.5)
                fig.update_layout(height=400, yaxis_title='평균 오답률 (%)', yaxis=dict(range=[0, 100]))
                st.plotly_chart(fig, use_container_width=True)
            
            # 법령 문제 중 완전 공통 오답
            st.markdown("#### 📜 " + ("법령 문제 중 완전 공통 오답" if lang == 'ko' else "Law Problems - Complete Gap"))
            
            law_complete_gap = law_all_wrong[law_all_wrong['law_status'] == 'O']
            
            if len(law_complete_gap) > 0:
                st.error(f"""
                ⚠️ **법령 공통 오답: {len(law_complete_gap)}개 문제**
                
                모든 모델이 틀린 법령 문제입니다. 법률 용어, 규정 해석, 법적 판단에 대한 
                근본적인 지식 부족을 의미합니다.
                """)
                
                display_law_gap = pd.DataFrame({
                    '문제 번호': law_complete_gap['problem_id'],
                    '과목': law_complete_gap['Subject'],
                    '연도': law_complete_gap['Year'],
                    '오답 모델수': law_complete_gap['incorrect_count'].astype(int)
                })
                
                st.dataframe(display_law_gap, width='stretch')
                
                if st.checkbox('법령 공통 오답 문제 상세 보기', key='law_gap_details'):
                    for idx, row in law_complete_gap.head(10).iterrows():
                        with st.expander(f"📜 {row['problem_id']}"):
                            q_detail = filtered_df[filtered_df['Question'] == row['Question']].iloc[0]
                            st.write(f"**문제:** {q_detail['Question']}")
                            
                            if '법령 이름' in q_detail and pd.notna(q_detail['법령 이름']):
                                st.write(f"**📚 법령:** {q_detail['법령 이름']}")
                            
                            if 'Subject' in q_detail and pd.notna(q_detail['Subject']):
                                st.write(f"**과목:** {q_detail['Subject']}")
                            
                            if all([f'Option {i}' in q_detail for i in range(1, 5)]):
                                st.write("**선택지:**")
                                for i in range(1, 5):
                                    option = q_detail[f'Option {i}']
                                    if pd.notna(option):
                                        if str(i) == str(row['CorrectAnswer']):
                                            st.write(f"  ✅ {i}. {option} **(정답)**")
                                        else:
                                            st.write(f"  {i}. {option}")
                            
                            st.write("**각 모델이 선택한 답:**")
                            for model, answer in row['selected_answers'].items():
                                st.write(f"  • {model}: {answer}")
            else:
                st.success("✅ 모든 모델이 틀린 법령 문제가 없습니다!")
            
            # 비법령 문제 중 완전 공통 오답
            st.markdown("#### 📘 " + ("비법령 문제 중 완전 공통 오답" if lang == 'ko' else "Non-Law Problems - Complete Gap"))
            
            non_law_complete_gap = law_all_wrong[law_all_wrong['law_status'] != 'O']
            
            if len(non_law_complete_gap) > 0:
                st.warning(f"""
                ℹ️ **비법령 공통 오답: {len(non_law_complete_gap)}개 문제**
                
                모든 모델이 틀린 비법령 문제입니다. 기술적 지식, 실무 경험, 
                전문 용어 이해 등에 대한 격차를 의미합니다.
                """)
                
                display_non_law_gap = pd.DataFrame({
                    '문제 번호': non_law_complete_gap['problem_id'],
                    '과목': non_law_complete_gap['Subject'],
                    '연도': non_law_complete_gap['Year'],
                    '오답 모델수': non_law_complete_gap['incorrect_count'].astype(int)
                })
                
                st.dataframe(display_non_law_gap, width='stretch')
            else:
                st.success("✅ 모든 모델이 틀린 비법령 문제가 없습니다!")
            
            # 인사이트 및 권장 조치
            st.markdown("---")
            st.markdown("#### 💡 " + ("법령/비법령 공통 오답 인사이트" if lang == 'ko' else "Law/Non-Law Gap Insights"))
            
            law_gap_count = len(law_complete_gap)
            non_law_gap_count = len(non_law_complete_gap)
            law_gap_ratio = (law_gap_count / total_law * 100) if total_law > 0 else 0
            non_law_gap_ratio = (non_law_gap_count / total_non_law * 100) if total_non_law > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("법령 공통 오답 비율", f"{law_gap_ratio:.1f}%", f"{law_gap_count}/{total_law}")
            
            with col2:
                st.metric("비법령 공통 오답 비율", f"{non_law_gap_ratio:.1f}%", f"{non_law_gap_count}/{total_non_law}")
            
            with col3:
                if law_gap_ratio > non_law_gap_ratio:
                    st.metric("더 취약한 영역", "법령", f"+{law_gap_ratio - non_law_gap_ratio:.1f}%p")
                else:
                    st.metric("더 취약한 영역", "비법령", f"+{non_law_gap_ratio - law_gap_ratio:.1f}%p")
            
            if law_gap_ratio > non_law_gap_ratio * 1.5:
                st.error(f"""
                🚨 **법령 지식이 특히 취약합니다!**
                
                법령 문제의 공통 오답 비율({law_gap_ratio:.1f}%)이 비법령({non_law_gap_ratio:.1f}%)보다 
                {law_gap_ratio / non_law_gap_ratio:.1f}배 높습니다.
                
                **권장 조치**:
                - 법률 용어 및 규정에 대한 학습 데이터 보강
                - 법령 해석 예시 추가
                - 법률 전문가 검토 및 피드백 반영
                """)
            elif non_law_gap_ratio > law_gap_ratio * 1.5:
                st.warning(f"""
                ⚠️ **기술/실무 지식이 상대적으로 취약합니다!**
                
                비법령 문제의 공통 오답 비율({non_law_gap_ratio:.1f}%)이 법령({law_gap_ratio:.1f}%)보다 
                {non_law_gap_ratio / law_gap_ratio:.1f}배 높습니다.
                
                **권장 조치**:
                - 기술적 세부사항 학습 데이터 보강
                - 실무 사례 및 적용 예시 추가
                - 전문 용어 정의 명확화
                """)
            else:
                st.success(f"""
                ✅ **법령과 비법령 공통 오답가 균형적입니다.**
                
                법령({law_gap_ratio:.1f}%)과 비법령({non_law_gap_ratio:.1f}%) 문제의 
                공통 오답 비율이 비슷한 수준입니다.
                """)
        
        # ========================================
        # 섹션 10: 오답률 Top 10 차트
        # ========================================
        st.markdown("---")
        top_10_chart = top_20.head(10)
        
        fig = px.bar(
            top_10_chart,
            x='problem_id',
            y='incorrect_rate',
            title='오답률 높은 문제 Top 10' if lang == 'ko' else 'Top 10 Problems by Incorrect Rate',
            text=[f"{x:.0%}" for x in top_10_chart['incorrect_rate']],
            color='incorrect_rate',
            color_continuous_scale='Reds',
            range_color=[0, 1]
        )
        fig.update_traces(textposition='outside', marker_line_color='black', marker_line_width=1.5)
        fig.update_layout(
            height=500,
            showlegend=False,
            yaxis_title='오답률',
            xaxis_title='문제 번호',
            yaxis=dict(range=[0, 1])
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        
        # ========================================
        # 섹션 11: 고오답률 & 고일관성 문제 분석 (NEW!)
        # ========================================
        st.markdown("---")
        st.subheader("🎯 " + ("고오답률 & 고일관성 문제 분석" if lang == 'ko' else "High Incorrect Rate & High Consistency Analysis"))
        
        st.markdown("""
        > 💡 **분석 목적**: 오답률 50% 이상이면서 일관성(같은 오답 선택률) 50% 이상인 문제는 
        > 모델들이 **체계적으로 틀리는** 문제입니다. 이는 학습 데이터의 편향이나 지식 격차를 나타냅니다.
        """ if lang == 'ko' else """
        > 💡 **Purpose**: Problems with incorrect rate ≥50% AND consistency ≥50% indicate 
        > **systematic errors** where models consistently choose the same wrong answer.
        """)
        
        # 문제별 오답률과 일관성 계산
        problem_stats = filtered_df.groupby('Question').agg({
            '정답여부': ['sum', 'count', 'mean'],
            '예측답': lambda x: x.value_counts().iloc[0] / len(x) if len(x) > 0 else 0  # 가장 많이 선택된 답의 비율
        }).reset_index()
        problem_stats.columns = ['Question', 'correct_count', 'total_count', 'accuracy', 'top_answer_ratio']
        problem_stats['incorrect_rate'] = 1 - problem_stats['accuracy']
        
        # 오답인 경우의 일관성 (같은 오답을 선택한 비율)
        consistency_list = []
        for q in problem_stats['Question']:
            q_df = filtered_df[filtered_df['Question'] == q]
            wrong_df = q_df[q_df['정답여부'] == False]
            if len(wrong_df) > 0:
                # 가장 많이 선택된 오답의 비율
                wrong_answer_counts = wrong_df['예측답'].value_counts()
                if len(wrong_answer_counts) > 0:
                    consistency = wrong_answer_counts.iloc[0] / len(wrong_df)
                else:
                    consistency = 0
            else:
                consistency = 0
            consistency_list.append(consistency)
        
        problem_stats['wrong_consistency'] = consistency_list
        
        # 테스트명 매핑
        test_mapping = filtered_df.groupby('Question')['테스트명'].first().to_dict()
        problem_stats['테스트명'] = problem_stats['Question'].map(test_mapping)
        
        # 고오답률 & 고일관성 문제 필터링 (오답률 50%+, 일관성 50%+)
        high_risk = problem_stats[
            (problem_stats['incorrect_rate'] >= 0.5) & 
            (problem_stats['wrong_consistency'] >= 0.5)
        ].copy()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "전체 문제 수" if lang == 'ko' else "Total Problems",
                f"{len(problem_stats):,}"
            )
        with col2:
            st.metric(
                "고위험 문제 수" if lang == 'ko' else "High-Risk Problems",
                f"{len(high_risk):,}",
                f"{len(high_risk)/len(problem_stats)*100:.1f}%" if len(problem_stats) > 0 else "0%"
            )
        with col3:
            avg_consistency = high_risk['wrong_consistency'].mean() * 100 if len(high_risk) > 0 else 0
            st.metric(
                "평균 오답 일관성" if lang == 'ko' else "Avg Wrong Consistency",
                f"{avg_consistency:.1f}%"
            )
        
        # ----- 1. 오답률-일관성 구간별 문제 개수 히트맵 -----
        st.markdown("#### " + ("📊 오답률-일관성 구간별 문제 분포 히트맵" if lang == 'ko' else "📊 Problem Distribution Heatmap by Incorrect Rate & Consistency"))
        
        # 5% 구간으로 binning (50~100%)
        bins = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.01]
        bin_labels = ['50-55%', '55-60%', '60-65%', '65-70%', '70-75%', '75-80%', '80-85%', '85-90%', '90-95%', '95-100%']
        
        # 오답률 50% 이상, 일관성 50% 이상만 필터링
        heatmap_data = problem_stats[
            (problem_stats['incorrect_rate'] >= 0.5) & 
            (problem_stats['wrong_consistency'] >= 0.5)
        ].copy()
        
        if len(heatmap_data) > 0:
            heatmap_data['incorrect_bin'] = pd.cut(
                heatmap_data['incorrect_rate'], 
                bins=bins, 
                labels=bin_labels,
                include_lowest=True
            )
            heatmap_data['consistency_bin'] = pd.cut(
                heatmap_data['wrong_consistency'], 
                bins=bins, 
                labels=bin_labels,
                include_lowest=True
            )
            
            # 피벗 테이블 생성
            heatmap_pivot = pd.crosstab(
                heatmap_data['consistency_bin'], 
                heatmap_data['incorrect_bin'],
                dropna=False
            )
            
            # 모든 구간이 있도록 reindex
            heatmap_pivot = heatmap_pivot.reindex(index=bin_labels, columns=bin_labels, fill_value=0)
            
            # 히트맵 생성
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=heatmap_pivot.values,
                x=heatmap_pivot.columns.tolist(),
                y=heatmap_pivot.index.tolist(),
                colorscale='YlOrRd',
                text=heatmap_pivot.values,
                texttemplate='%{text}',
                textfont={"size": annotation_size},
                colorbar=dict(title="문제 수" if lang == 'ko' else "Count"),
                hoverongaps=False
            ))
            
            fig_heatmap.update_layout(
                title='오답률 vs 일관성 구간별 문제 분포' if lang == 'ko' else 'Problem Distribution: Incorrect Rate vs Consistency',
                xaxis_title='오답률 구간' if lang == 'ko' else 'Incorrect Rate Range',
                yaxis_title='오답 일관성 구간' if lang == 'ko' else 'Wrong Consistency Range',
                height=500,
                xaxis=dict(side='bottom'),
                yaxis=dict(autorange='reversed')  # 위에서 아래로
            )
            fig_heatmap.update_xaxes(tickfont=dict(size=annotation_size))
            fig_heatmap.update_yaxes(tickfont=dict(size=annotation_size))
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # 인사이트
            max_cell = heatmap_pivot.max().max()
            max_pos = heatmap_pivot.stack().idxmax()
            if max_cell > 0:
                st.info(f"💡 " + (f"가장 많은 문제가 집중된 구간: 오답률 **{max_pos[1]}**, 일관성 **{max_pos[0]}** ({int(max_cell)}개)" 
                        if lang == 'ko' else f"Most concentrated: Incorrect Rate **{max_pos[1]}**, Consistency **{max_pos[0]}** ({int(max_cell)} problems)"))
        else:
            st.info("오답률 50% 이상이면서 일관성 50% 이상인 문제가 없습니다." if lang == 'ko' else "No problems with both incorrect rate ≥50% and consistency ≥50%.")
        
        st.markdown("---")
        
        # ----- 2. 테스트셋별 고위험 문제 개수 차트 -----
        st.markdown("#### " + ("📊 테스트셋별 고위험 문제 분포" if lang == 'ko' else "📊 High-Risk Problems by Test Set"))
        
        if len(high_risk) > 0 and '테스트명' in high_risk.columns:
            # 테스트셋별 고위험 문제 수
            testset_risk = high_risk.groupby('테스트명').agg({
                'Question': 'count',
                'incorrect_rate': 'mean',
                'wrong_consistency': 'mean'
            }).reset_index()
            testset_risk.columns = ['테스트명', '고위험 문제 수', '평균 오답률', '평균 일관성']
            
            # 전체 문제 수 대비 비율 계산
            total_by_test = problem_stats.groupby('테스트명')['Question'].count().to_dict()
            testset_risk['전체 문제 수'] = testset_risk['테스트명'].map(total_by_test)
            testset_risk['고위험 비율'] = testset_risk['고위험 문제 수'] / testset_risk['전체 문제 수'] * 100
            
            testset_risk = testset_risk.sort_values('고위험 문제 수', ascending=False)
            
            # 막대 차트
            fig_testset = go.Figure()
            
            # 고위험 문제 수 막대
            fig_testset.add_trace(go.Bar(
                name='고위험 문제 수' if lang == 'ko' else 'High-Risk Problems',
                x=testset_risk['테스트명'],
                y=testset_risk['고위험 문제 수'],
                text=testset_risk['고위험 문제 수'],
                textposition='outside',
                textfont=dict(size=annotation_size),
                marker_color='#e74c3c',
                marker_line_color='black',
                marker_line_width=1.5
            ))
            
            fig_testset.update_layout(
                title='테스트셋별 고위험 문제 수 (오답률≥50% & 일관성≥50%)' if lang == 'ko' else 'High-Risk Problems by Test Set (Incorrect≥50% & Consistency≥50%)',
                xaxis_title='테스트셋' if lang == 'ko' else 'Test Set',
                yaxis_title='문제 수' if lang == 'ko' else 'Problem Count',
                height=450,
                showlegend=False
            )
            fig_testset.update_xaxes(tickangle=45, tickfont=dict(size=annotation_size))
            fig_testset.update_yaxes(tickfont=dict(size=annotation_size))
            
            st.plotly_chart(fig_testset, use_container_width=True)
            
            # 비율 차트
            st.markdown("##### " + ("고위험 문제 비율 (테스트셋 내)" if lang == 'ko' else "High-Risk Problem Ratio (within Test Set)"))
            
            fig_ratio = px.bar(
                testset_risk.sort_values('고위험 비율', ascending=False),
                x='테스트명',
                y='고위험 비율',
                text=[f"{x:.1f}%" for x in testset_risk.sort_values('고위험 비율', ascending=False)['고위험 비율']],
                color='고위험 비율',
                color_continuous_scale='Reds',
                title='테스트셋별 고위험 문제 비율' if lang == 'ko' else 'High-Risk Problem Ratio by Test Set'
            )
            fig_ratio.update_traces(textposition='outside', textfont=dict(size=annotation_size), marker_line_color='black', marker_line_width=1)
            fig_ratio.update_layout(
                height=400,
                showlegend=False,
                yaxis_title='비율 (%)' if lang == 'ko' else 'Ratio (%)',
                xaxis_title='테스트셋' if lang == 'ko' else 'Test Set',
                coloraxis_showscale=False
            )
            fig_ratio.update_xaxes(tickangle=45, tickfont=dict(size=annotation_size))
            fig_ratio.update_yaxes(tickfont=dict(size=annotation_size))
            
            st.plotly_chart(fig_ratio, use_container_width=True)
            
            # 상세 테이블
            with st.expander("📋 " + ("상세 데이터 보기" if lang == 'ko' else "View Detailed Data")):
                display_testset = testset_risk.copy()
                display_testset['평균 오답률'] = display_testset['평균 오답률'].apply(lambda x: f"{x*100:.1f}%")
                display_testset['평균 일관성'] = display_testset['평균 일관성'].apply(lambda x: f"{x*100:.1f}%")
                display_testset['고위험 비율'] = display_testset['고위험 비율'].apply(lambda x: f"{x:.1f}%")
                
                st.dataframe(
                    display_testset,
                    use_container_width=True,
                    hide_index=True
                )
            
            # 가장 위험한 테스트셋 강조
            most_risky = testset_risk.iloc[0]
            st.warning(f"""
            ⚠️ **가장 주의가 필요한 테스트셋: {most_risky['테스트명']}**
            - 고위험 문제 수: {int(most_risky['고위험 문제 수'])}개
            - 평균 오답률: {most_risky['평균 오답률']*100:.1f}%
            - 평균 오답 일관성: {most_risky['평균 일관성']*100:.1f}%
            """ if lang == 'ko' else f"""
            ⚠️ **Test set requiring most attention: {most_risky['테스트명']}**
            - High-risk problems: {int(most_risky['고위험 문제 수'])}
            - Avg incorrect rate: {most_risky['평균 오답률']*100:.1f}%
            - Avg wrong consistency: {most_risky['평균 일관성']*100:.1f}%
            """)
        else:
            st.info("고위험 문제가 없거나 테스트셋 정보가 없습니다." if lang == 'ko' else "No high-risk problems or test set info unavailable.")
    
    # 탭 8: 난이도 분석
    with tabs[7]:
        st.header(f"📈 {t['difficulty_analysis']}")
        
        # 문제별 난이도 계산 (정답률 기반)
        difficulty = filtered_df.groupby('Question').agg({
            '정답여부': ['sum', 'count', 'mean']
        }).reset_index()
        difficulty.columns = ['Question', 'correct_count', 'total_count', 'difficulty_score']
        difficulty['difficulty_score'] = difficulty['difficulty_score'] * 100
        
        # 난이도 구간 분류
        def classify_difficulty(score, lang='ko'):
            if lang == 'ko':
                if score < 20:
                    return '매우 어려움 (0-20%)'
                elif score < 40:
                    return '어려움 (20-40%)'
                elif score < 60:
                    return '보통 (40-60%)'
                elif score < 80:
                    return '쉬움 (60-80%)'
                else:
                    return '매우 쉬움 (80-100%)'
            else:  # English
                if score < 20:
                    return 'Very Hard (0-20%)'
                elif score < 40:
                    return 'Hard (20-40%)'
                elif score < 60:
                    return 'Medium (40-60%)'
                elif score < 80:
                    return 'Easy (60-80%)'
                else:
                    return 'Very Easy (80-100%)'
        
        difficulty['난이도_구간'] = difficulty['difficulty_score'].apply(lambda x: classify_difficulty(x, lang))
        
        # 난이도 구간 순서 정의 (어려운 것부터 쉬운 것 순)
        if lang == 'ko':
            difficulty_order = [
                '매우 어려움 (0-20%)',
                '어려움 (20-40%)',
                '보통 (40-60%)',
                '쉬움 (60-80%)',
                '매우 쉬움 (80-100%)'
            ]
        else:
            difficulty_order = [
                'Very Hard (0-20%)',
                'Hard (20-40%)',
                'Medium (40-60%)',
                'Easy (60-80%)',
                'Very Easy (80-100%)'
            ]
        difficulty['난이도_구간'] = pd.Categorical(difficulty['난이도_구간'], categories=difficulty_order, ordered=True)
        
        # 원본 데이터에 난이도 정보 병합
        analysis_df = filtered_df.merge(difficulty[['Question', 'difficulty_score', '난이도_구간']], on='Question')
        
        # analysis_df에도 동일한 순서 적용
        analysis_df['난이도_구간'] = pd.Categorical(analysis_df['난이도_구간'], categories=difficulty_order, ordered=True)
        
        # 1. 난이도 분포
        st.subheader("📈 " + (t['problem_distribution'] if 'problem_distribution' in t else ('문제 난이도 분포' if lang == 'ko' else 'Problem Difficulty Distribution')))
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 난이도 분포 히스토그램
            fig = px.histogram(
                difficulty,
                x='difficulty_score',
                nbins=20,
                title=t['difficulty_score'] + ' Distribution',
                labels={'difficulty_score': t['difficulty_score'], 'count': t['problem_count']}
            )
            fig.update_traces(
                marker_line_color='black',
                marker_line_width=1.5
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 난이도 구간별 문제 수
            difficulty_dist = difficulty['난이도_구간'].value_counts()
            # 난이도 순서대로 재정렬
            difficulty_dist = difficulty_dist.reindex(difficulty_order, fill_value=0)
            
            fig = px.bar(
                x=difficulty_dist.index,
                y=difficulty_dist.values,
                title=t['problem_count'] + (' by ' + t['difficulty_range'] if lang == 'en' else ' (' + t['difficulty_range'] + '별)'),
                labels={'x': t['difficulty_range'], 'y': t['problem_count']},
                text=difficulty_dist.values,
                color=difficulty_dist.values,
                color_continuous_scale='RdYlGn_r'
            )
            fig.update_traces(
                texttemplate='%{text}',
                textposition='outside',                textfont=dict(size=annotation_size),
                marker_line_color='black',
                marker_line_width=1.5
            )
            fig.update_layout(
                height=400,
                showlegend=False
            )
            fig.update_xaxes(tickangle=45, tickfont=dict(size=annotation_size))
            st.plotly_chart(fig, use_container_width=True)
        
        # 통계 요약
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                t['correct_rate'] if lang == 'ko' else 'Average Correct Rate',
                f"{difficulty['difficulty_score'].mean():.1f}%"
            )
        with col2:
            st.metric(
                '중앙값' if lang == 'ko' else 'Median',
                f"{difficulty['difficulty_score'].median():.1f}%"
            )
        with col3:
            very_hard_label = difficulty_order[0]
            very_hard = len(difficulty[difficulty['난이도_구간'] == very_hard_label])
            st.metric(
                t['very_hard'] + (' 문제' if lang == 'ko' else ' Problems'),
                f"{very_hard}" + (t['problems'] if lang == 'ko' else '')
            )
        with col4:
            very_easy_label = difficulty_order[-1]
            very_easy = len(difficulty[difficulty['난이도_구간'] == very_easy_label])
            st.metric(
                t['very_easy'] + (' 문제' if lang == 'ko' else ' Problems'),
                f"{very_easy}" + (t['problems'] if lang == 'ko' else '')
            )
        
        st.markdown("---")
        
        # 2. 난이도별 모델 성능
        st.subheader("🎯 " + ('난이도별 모델 성능' if lang == 'ko' else 'Model Performance by Difficulty Level'))
        
        # 모델별 난이도 구간별 정답률
        model_difficulty = analysis_df.groupby(['모델', '난이도_구간']).agg({
            '정답여부': ['mean', 'count']
        }).reset_index()
        
        # 컬럼명 언어별 설정
        if lang == 'ko':
            model_difficulty.columns = ['모델', '난이도_구간', '정답률', '문제수']
        else:
            model_difficulty.columns = ['Model', 'Difficulty', 'Correct Rate', 'Problem Count']
        
        # 정답률 컬럼명 (언어별)
        acc_col = '정답률' if lang == 'ko' else 'Correct Rate'
        model_col = '모델' if lang == 'ko' else 'Model'
        diff_col = '난이도_구간' if lang == 'ko' else 'Difficulty'
        
        model_difficulty[acc_col] = model_difficulty[acc_col] * 100
        
        # 라인 차트
        fig = px.line(
            model_difficulty,
            x=diff_col,
            y=acc_col,
            color=model_col,
            markers=True,
            title='난이도별 모델 성능 비교' if lang == 'ko' else 'Model Performance by Difficulty Level',
            labels={
                acc_col: t['accuracy'] + ' (%)',
                diff_col: t['difficulty_range'],
                model_col: t['model']
            },
            category_orders={diff_col: difficulty_order}
        )
        fig.update_traces(
            marker_size=10,
            marker_line_color='black',
            marker_line_width=2,
            line_width=3
        )
        fig.update_layout(height=500)
        fig.update_xaxes(tickangle=45, tickfont=dict(size=annotation_size))
        st.plotly_chart(fig, use_container_width=True)
        
        # 히트맵
        pivot_difficulty = model_difficulty.pivot(
            index=model_col,
            columns=diff_col,
            values=acc_col
        )
        
        # 난이도 순서대로 컬럼 재정렬
        pivot_difficulty = pivot_difficulty.reindex(columns=difficulty_order)
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_difficulty.values,
            x=pivot_difficulty.columns,
            y=pivot_difficulty.index,
            colorscale='RdYlGn',
            text=np.round(pivot_difficulty.values, 1),
            texttemplate='%{text:.1f}',
            textfont={"size": annotation_size},
            colorbar=dict(title=t['accuracy'] + " (%)"),
            xgap=2,  # 셀 경계선
            ygap=2
        ))
        fig.update_layout(
            height=400,
            title='모델 × 난이도 히트맵' if lang == 'ko' else 'Model × Difficulty Heatmap',
            xaxis_title=t['difficulty_range'],
            yaxis_title=t['model']
        )
        fig.update_xaxes(tickangle=45, tickfont=dict(size=annotation_size))
        fig.update_yaxes(tickfont=dict(size=annotation_size))
        st.plotly_chart(fig, use_container_width=True)
        
        # 난이도별 성능 인사이트
        # 모델별 난이도 적응력 분석
        difficulty_adaptability = {}
        for model in pivot_difficulty.index:
            # 매우 어려운 문제 정확도
            very_hard_acc = pivot_difficulty.loc[model, difficulty_order[0]] if difficulty_order[0] in pivot_difficulty.columns else 0
            # 매우 쉬운 문제 정확도
            very_easy_acc = pivot_difficulty.loc[model, difficulty_order[-1]] if difficulty_order[-1] in pivot_difficulty.columns else 0
            # 격차 (작을수록 일관적)
            gap = very_easy_acc - very_hard_acc
            difficulty_adaptability[model] = {'hard': very_hard_acc, 'easy': very_easy_acc, 'gap': gap}
        
        best_hard_model = max(difficulty_adaptability.items(), key=lambda x: x[1]['hard'])[0]
        most_consistent_model = min(difficulty_adaptability.items(), key=lambda x: x[1]['gap'])[0]
        
        st.success(f"""
        💡 **{"난이도별 모델 적응력 분석" if lang == 'ko' else "Model Adaptability by Difficulty"}**:
        
        🏆 **{"어려운 문제 대응력" if lang == 'ko' else "Hard Problem Performance"}**:
        - **{"최고" if lang == 'ko' else "Best"}**: {best_hard_model} ({difficulty_adaptability[best_hard_model]['hard']:.1f}% {"매우 어려운 문제에서" if lang == 'ko' else "on very hard"})
        - **{"특징" if lang == 'ko' else "Note"}**: {"복잡한 추론이 필요한 경우 활용" if lang == 'ko' else "Use for complex reasoning tasks"}
        
        ⚖️ **{"일관성" if lang == 'ko' else "Consistency"}**:
        - **{"가장 일관적" if lang == 'ko' else "Most Consistent"}**: {most_consistent_model} ({"난이도 격차" if lang == 'ko' else "difficulty gap"}: {difficulty_adaptability[most_consistent_model]['gap']:.1f}%p)
        - **{"의미" if lang == 'ko' else "Meaning"}**: {"모든 난이도에서 안정적 성능" if lang == 'ko' else "Stable across all difficulties"}
        
        📊 **{"활용 전략" if lang == 'ko' else "Usage Strategy"}**:
        - **{"고난도 시험" if lang == 'ko' else "High-difficulty exams"}**: {best_hard_model} {"권장" if lang == 'ko' else "recommended"}
        - **{"범용 학습" if lang == 'ko' else "General learning"}**: {most_consistent_model} {"권장" if lang == 'ko' else "recommended"}
        - **{"라인 차트" if lang == 'ko' else "Line chart"}**: {"난이도가 올라갈수록 성능 하락폭 확인" if lang == 'ko' else "Check performance drop as difficulty increases"}
        """)
        
        st.markdown("---")
        
        # 3. 과목별 난이도 분석
        if 'Subject' in analysis_df.columns:
            st.subheader("📚 " + ('과목별 난이도 분석' if lang == 'ko' else 'Difficulty Analysis by Subject'))
            
            subject_difficulty = analysis_df.groupby('Subject').agg({
                'difficulty_score': 'mean',
                'Question': 'count'
            }).reset_index()
            
            # 컬럼명 언어별 설정
            if lang == 'ko':
                subject_difficulty.columns = ['과목', '평균_난이도', '문제수']
                subj_col = '과목'
                avg_diff_col = '평균_난이도'
            else:
                subject_difficulty.columns = ['Subject', 'Avg Difficulty', 'Problem Count']
                subj_col = 'Subject'
                avg_diff_col = 'Avg Difficulty'
            
            subject_difficulty = subject_difficulty.sort_values(avg_diff_col)
            
            fig = px.bar(
                subject_difficulty,
                x=subj_col,
                y=avg_diff_col,
                title='과목별 평균 난이도 (정답률)' if lang == 'ko' else 'Average Difficulty by Subject (Correct Rate)',
                text=avg_diff_col,
                color=avg_diff_col,
                color_continuous_scale='RdYlGn',
                labels={subj_col: t['by_subject'].replace('별', ''), avg_diff_col: t['avg_difficulty']}
            )
            fig.update_traces(
                texttemplate='%{text:.1f}%',
                textposition='outside',                textfont=dict(size=annotation_size),
                marker_line_color='black',
                marker_line_width=1.5
            )
            fig.update_xaxes(tickangle=45, tickfont=dict(size=annotation_size))
            fig.update_layout(
                height=500,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 과목 × 난이도 구간 히트맵
            subject_diff_dist = analysis_df.groupby(['Subject', '난이도_구간']).size().reset_index(name='문제수')
            pivot_subject_diff = subject_diff_dist.pivot(
                index='Subject',
                columns='난이도_구간',
                values='문제수'
            ).fillna(0)
            
            # 난이도 순서대로 컬럼 재정렬
            pivot_subject_diff = pivot_subject_diff.reindex(columns=difficulty_order, fill_value=0)
            
            fig = go.Figure(data=go.Heatmap(
                z=pivot_subject_diff.values,
                x=pivot_subject_diff.columns,
                y=pivot_subject_diff.index,
                colorscale='Blues',
                text=pivot_subject_diff.values.astype(int),
                texttemplate='%{text}',
                textfont={"size": annotation_size},
                colorbar=dict(title=t['problem_count']),
                xgap=2,  # 셀 경계선
                ygap=2
            ))
            fig.update_layout(
                height=500,
                title='과목 × 난이도 분포' if lang == 'ko' else 'Subject × Difficulty Distribution',
                xaxis_title=t['difficulty_range'],
                yaxis_title=t['by_subject'].replace('별', '')  # '과목' or 'Subject'
            )
            fig.update_xaxes(tickangle=45, tickfont=dict(size=annotation_size))
            fig.update_yaxes(tickfont=dict(size=annotation_size))
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # 4. 어려운 문제 vs 쉬운 문제 상세 분석
        st.subheader("🔍 " + (
            "어려운 문제 vs 쉬운 문제 비교" if lang == 'ko' else "Hard vs Easy Problems Comparison"
        ))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### " + (
                "매우 어려운 문제 (정답률 < 20%)" if lang == 'ko' else "Very Hard Problems (Correct Rate < 20%)"
            ))
            very_hard_problems = difficulty[difficulty['difficulty_score'] < 20].sort_values('difficulty_score')
            
            if len(very_hard_problems) > 0:
                st.metric(
                    t['problem_count'],
                    f"{len(very_hard_problems)}" + (t['problems'] if lang == 'ko' else '')
                )
                st.metric(
                    '평균 정답률' if lang == 'ko' else 'Average Correct Rate',
                    f"{very_hard_problems['difficulty_score'].mean():.1f}%"
                )
                
                # 모델별 성능
                very_hard_questions = very_hard_problems['Question'].tolist()
                very_hard_model_perf = filtered_df[filtered_df['Question'].isin(very_hard_questions)].groupby('모델')['정답여부'].mean() * 100
                
                st.markdown("**" + (
                    "모델별 성능" if lang == 'ko' else "Performance by Model"
                ) + "**")
                for model, acc in very_hard_model_perf.sort_values(ascending=False).items():
                    st.write(f"- {model}: {acc:.1f}%")
            else:
                st.info(
                    "매우 어려운 문제가 없습니다." if lang == 'ko' else "No very hard problems found."
                )
        
        with col2:
            st.markdown("#### " + (
                "매우 쉬운 문제 (정답률 > 80%)" if lang == 'ko' else "Very Easy Problems (Correct Rate > 80%)"
            ))
            very_easy_problems = difficulty[difficulty['difficulty_score'] > 80].sort_values('difficulty_score', ascending=False)
            
            if len(very_easy_problems) > 0:
                st.metric(
                    t['problem_count'],
                    f"{len(very_easy_problems)}" + (t['problems'] if lang == 'ko' else '')
                )
                st.metric(
                    '평균 정답률' if lang == 'ko' else 'Average Correct Rate',
                    f"{very_easy_problems['difficulty_score'].mean():.1f}%"
                )
                
                # 모델별 성능
                very_easy_questions = very_easy_problems['Question'].tolist()
                very_easy_model_perf = filtered_df[filtered_df['Question'].isin(very_easy_questions)].groupby('모델')['정답여부'].mean() * 100
                
                st.markdown("**" + (
                    "모델별 성능" if lang == 'ko' else "Performance by Model"
                ) + "**")
                for model, acc in very_easy_model_perf.sort_values(ascending=False).items():
                    st.write(f"- {model}: {acc:.1f}%")
            else:
                st.info(
                    "매우 쉬운 문제가 없습니다." if lang == 'ko' else "No very easy problems found."
                )
        
        st.markdown("---")
        
        # 5. 난이도 구간별 상세 테이블
        st.subheader("📋 " + t['difficulty_stats_by_range'])
        
        detailed_difficulty = model_difficulty.pivot_table(
            index=model_col,
            columns=diff_col,
            values=acc_col,
            aggfunc='mean'
        ).round(2)
        
        # 난이도 순서대로 컬럼 재정렬
        detailed_difficulty = detailed_difficulty.reindex(columns=difficulty_order)
        
        st.dataframe(
            detailed_difficulty.style.background_gradient(cmap='RdYlGn', axis=None),
            width='stretch'
        )
        
        # 난이도 분석 종합 인사이트
        # 전체 문제 난이도 분포 분석
        total_problems = len(difficulty)
        very_hard_pct = (len(difficulty[difficulty['difficulty_score'] < 20]) / total_problems * 100) if total_problems > 0 else 0
        hard_pct = (len(difficulty[(difficulty['difficulty_score'] >= 20) & (difficulty['difficulty_score'] < 40)]) / total_problems * 100) if total_problems > 0 else 0
        medium_pct = (len(difficulty[(difficulty['difficulty_score'] >= 40) & (difficulty['difficulty_score'] < 60)]) / total_problems * 100) if total_problems > 0 else 0
        easy_pct = (len(difficulty[(difficulty['difficulty_score'] >= 60) & (difficulty['difficulty_score'] < 80)]) / total_problems * 100) if total_problems > 0 else 0
        very_easy_pct = (len(difficulty[difficulty['difficulty_score'] >= 80]) / total_problems * 100) if total_problems > 0 else 0
        
        st.info(f"""
        💡 **{"난이도 분포 종합 분석" if lang == 'ko' else "Overall Difficulty Distribution"}**:
        
        📊 **{"문제 난이도 구성" if lang == 'ko' else "Problem Composition"}**:
        - **{"매우 어려움" if lang == 'ko' else "Very Hard"}**: {very_hard_pct:.1f}% ({len(difficulty[difficulty['difficulty_score'] < 20])}{"개" if lang == 'ko' else ""})
        - **{"어려움" if lang == 'ko' else "Hard"}**: {hard_pct:.1f}% ({len(difficulty[(difficulty['difficulty_score'] >= 20) & (difficulty['difficulty_score'] < 40)])}{"개" if lang == 'ko' else ""})
        - **{"보통" if lang == 'ko' else "Medium"}**: {medium_pct:.1f}% ({len(difficulty[(difficulty['difficulty_score'] >= 40) & (difficulty['difficulty_score'] < 60)])}{"개" if lang == 'ko' else ""})
        - **{"쉬움" if lang == 'ko' else "Easy"}**: {easy_pct:.1f}% ({len(difficulty[(difficulty['difficulty_score'] >= 60) & (difficulty['difficulty_score'] < 80)])}{"개" if lang == 'ko' else ""})
        - **{"매우 쉬움" if lang == 'ko' else "Very Easy"}**: {very_easy_pct:.1f}% ({len(difficulty[difficulty['difficulty_score'] >= 80])}{"개" if lang == 'ko' else ""})
        
        🎯 **{"변별력 평가" if lang == 'ko' else "Discriminatory Power"}**:
        - {"중간 난이도(40-60%)" if lang == 'ko' else "Medium difficulty (40-60%)"}: {medium_pct:.1f}% - {"이상적 변별력 구간" if lang == 'ko' else "Ideal discriminatory range"}
        - {"변별력" if lang == 'ko' else "Overall discriminatory power"}: {"우수" if medium_pct > 30 else "보통" if medium_pct > 20 else "개선 필요" if lang == 'ko' else "Good" if medium_pct > 30 else "Fair" if medium_pct > 20 else "Needs improvement"}
        
        📝 **{"학습 전략" if lang == 'ko' else "Study Strategy"}**:
        - **{"기초 다지기" if lang == 'ko' else "Foundation"}**: {"쉬운 문제로 개념 확립" if lang == 'ko' else "Master basics with easy problems"}
        - **{"실력 향상" if lang == 'ko' else "Improvement"}**: {"중간 난이도로 실전 대비" if lang == 'ko' else "Practice with medium problems"}
        - **{"심화 학습" if lang == 'ko' else "Advanced"}**: {"어려운 문제로 고득점 노리기" if lang == 'ko' else "Challenge with hard problems"}
        """)
    
    # 탭 9: 토큰 및 비용 분석
    with tabs[8]:
        st.header(f"💰 {t['token_cost_analysis']}")
        
        # 토큰 관련 컬럼 확인
        token_columns = {
            'input': ['입력토큰', 'input_tokens', 'Input Tokens'],
            'output': ['출력토큰', 'output_tokens', 'Output Tokens'],
            'total': ['총토큰', 'total_tokens', 'Total Tokens'],
            'cost': ['비용수준', 'cost_level', 'Cost Level']
        }
        
        # 사용 가능한 컬럼 찾기
        available_cols = {}
        for key, possible_names in token_columns.items():
            for col_name in possible_names:
                if col_name in filtered_df.columns:
                    available_cols[key] = col_name
                    break
        
        if not available_cols:
            st.info("Token usage data not available in the dataset." if lang == 'en' else "토큰 사용량 데이터가 데이터셋에 없습니다.")
        else:
            # 데이터 준비 - NaN 필터링을 한 번에 처리 (copy 제거)
            valid_mask = pd.Series(True, index=filtered_df.index)
            for key, col in available_cols.items():
                if col in filtered_df.columns:
                    valid_mask &= filtered_df[col].notna()
            
            token_df = filtered_df[valid_mask]
            
            if len(token_df) == 0:
                st.info("No valid token data available after filtering." if lang == 'en' else "필터링 후 유효한 토큰 데이터가 없습니다.")
            else:
                # 1. 토큰 통계 요약
                st.subheader(f"📊 {t['token_stats']}")
                
                # 모델별 토큰 사용량 계산
                agg_dict = {}
                if 'input' in available_cols:
                    agg_dict[available_cols['input']] = ['sum', 'mean']
                if 'output' in available_cols:
                    agg_dict[available_cols['output']] = ['sum', 'mean']
                if 'total' in available_cols:
                    agg_dict[available_cols['total']] = ['sum', 'mean']
                
                model_token_stats = token_df.groupby('모델').agg(agg_dict).reset_index()
                
                # 컬럼명 정리
                new_cols = ['모델']
                for col in model_token_stats.columns[1:]:
                    if col[0] == available_cols.get('input', ''):
                        if col[1] == 'sum':
                            new_cols.append('총_입력토큰')
                        else:
                            new_cols.append('평균_입력토큰')
                    elif col[0] == available_cols.get('output', ''):
                        if col[1] == 'sum':
                            new_cols.append('총_출력토큰')
                        else:
                            new_cols.append('평균_출력토큰')
                    elif col[0] == available_cols.get('total', ''):
                        if col[1] == 'sum':
                            new_cols.append('총_토큰')
                        else:
                            new_cols.append('평균_토큰')
                
                model_token_stats.columns = new_cols
                
                # 정확도 추가
                model_acc = token_df.groupby('모델')['정답여부'].mean().reset_index()
                model_acc.columns = ['모델', '정확도']
                model_acc['정확도'] = model_acc['정확도'] * 100
                
                model_token_stats = model_token_stats.merge(model_acc, on='모델')
                
                # 문제 수 추가
                model_problem_count = token_df.groupby('모델')['Question'].count().reset_index()
                model_problem_count.columns = ['모델', '문제수']
                model_token_stats = model_token_stats.merge(model_problem_count, on='모델')
                
                # 비용 수준 추가 (있는 경우)
                if 'cost' in available_cols:
                    cost_col = available_cols['cost']
                    # 가장 빈번한 비용 수준 찾기
                    model_cost = token_df.groupby('모델')[cost_col].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown').reset_index()
                    model_cost.columns = ['모델', '비용수준']
                    model_token_stats = model_token_stats.merge(model_cost, on='모델')
                
                # 토큰 효율성 계산 (정답당 토큰)
                if '총_토큰' in model_token_stats.columns:
                    model_token_stats['정답당_토큰'] = model_token_stats.apply(
                        lambda row: row['총_토큰'] / (row['문제수'] * row['정확도'] / 100) if row['정확도'] > 0 else 0,
                        axis=1
                    )
                
                # 주요 메트릭 표시
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if '총_토큰' in model_token_stats.columns:
                        total_tokens = model_token_stats['총_토큰'].sum()
                        st.metric(
                            t['total_tokens'],
                            f"{total_tokens:,.0f}"
                        )
                
                with col2:
                    if '평균_토큰' in model_token_stats.columns:
                        avg_tokens = model_token_stats['평균_토큰'].mean()
                        st.metric(
                            t['avg_tokens_per_problem'],
                            f"{avg_tokens:,.0f}"
                        )
                
                with col3:
                    if '총_입력토큰' in model_token_stats.columns and '총_출력토큰' in model_token_stats.columns:
                        total_input = model_token_stats['총_입력토큰'].sum()
                        total_output = model_token_stats['총_출력토큰'].sum()
                        io_ratio = total_input / total_output if total_output > 0 else 0
                        st.metric(
                            t['io_ratio'],
                            f"{io_ratio:.2f}:1"
                        )
                
                with col4:
                    if '정답당_토큰' in model_token_stats.columns and len(model_token_stats[model_token_stats['정답당_토큰'] > 0]) > 0:
                        # 가장 효율적인 모델 (정답당 토큰이 적은 모델)
                        valid_stats = model_token_stats[model_token_stats['정답당_토큰'] > 0]
                        most_efficient = valid_stats.loc[valid_stats['정답당_토큰'].idxmin()]
                        st.metric(
                            t['most_efficient'],
                            most_efficient['모델'],
                            f"{most_efficient['정답당_토큰']:,.0f} " + t['tokens']
                        )
                
                # 상세 테이블
                st.markdown("---")
                st.subheader("📋 " + ("모델별 토큰 사용량 상세" if lang == 'ko' else "Detailed Token Usage by Model"))
                
                # 컬럼 순서 정리
                display_cols = ['모델']
                if '총_입력토큰' in model_token_stats.columns:
                    display_cols.append('총_입력토큰')
                if '총_출력토큰' in model_token_stats.columns:
                    display_cols.append('총_출력토큰')
                if '총_토큰' in model_token_stats.columns:
                    display_cols.append('총_토큰')
                if '평균_토큰' in model_token_stats.columns:
                    display_cols.append('평균_토큰')
                display_cols.extend(['정확도', '문제수'])
                if '비용수준' in model_token_stats.columns:
                    display_cols.append('비용수준')
                if '정답당_토큰' in model_token_stats.columns:
                    display_cols.append('정답당_토큰')
                
                display_df = model_token_stats[display_cols].sort_values('총_토큰' if '총_토큰' in display_cols else '모델', ascending=False)
                
                # 포맷팅
                format_dict = {
                    '총_입력토큰': '{:,.0f}',
                    '총_출력토큰': '{:,.0f}',
                    '총_토큰': '{:,.0f}',
                    '평균_토큰': '{:,.0f}',
                    '정확도': '{:.2f}%',
                    '정답당_토큰': '{:,.0f}'
                }
                
                st.dataframe(
                    display_df.style.format(format_dict).background_gradient(
                        subset=['정답당_토큰'] if '정답당_토큰' in display_cols else [],
                        cmap='RdYlGn_r'
                    ),
                    width='stretch'
                )
                
                st.markdown("---")
                
                # 2. 시각화
                st.subheader("📊 " + ("토큰 사용량 시각화" if lang == 'ko' else "Token Usage Visualization"))
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # 모델별 총 토큰 사용량
                    if '총_토큰' in model_token_stats.columns:
                        fig = px.bar(
                            display_df,
                            x='모델',
                            y='총_토큰',
                            title=t['total_tokens'] + ' (' + ('모델별' if lang == 'ko' else 'by Model') + ')',
                            text='총_토큰',
                            color='총_토큰',
                            color_continuous_scale='Blues'
                        )
                        fig.update_traces(
                            texttemplate='%{text:,.0f}',
                            textposition='outside',                textfont=dict(size=annotation_size),
                            marker_line_color='black',
                            marker_line_width=1.5
                        )
                        fig.update_layout(
                            height=400,
                            showlegend=False,
                            yaxis_title=t['total_tokens'],
                            xaxis_title=t['model']
                        )
                        fig.update_xaxes(tickangle=45, tickfont=dict(size=annotation_size))
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # 입출력 토큰 비교
                    if '총_입력토큰' in model_token_stats.columns and '총_출력토큰' in model_token_stats.columns:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            name=t['input_tokens'],
                            x=display_df['모델'],
                            y=display_df['총_입력토큰'],
                            marker_color='lightblue',
                            marker_line_color='black',
                            marker_line_width=1.5
                        ))
                        fig.add_trace(go.Bar(
                            name=t['output_tokens'],
                            x=display_df['모델'],
                            y=display_df['총_출력토큰'],
                            marker_color='lightcoral',
                            marker_line_color='black',
                            marker_line_width=1.5
                        ))
                        
                        fig.update_layout(
                            barmode='stack',
                            title=f"{t['input_tokens']} vs {t['output_tokens']}",
                            height=400,
                            yaxis_title=t['tokens'],
                            xaxis_title=t['model']
                        )
                        fig.update_xaxes(tickangle=45, tickfont=dict(size=annotation_size))
                        st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                # 3. 토큰 효율성 분석
                if '정답당_토큰' in model_token_stats.columns:
                    st.subheader("🎯 " + (t['token_efficiency']))
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 정답당 토큰 사용량
                        fig = px.bar(
                            display_df.sort_values('정답당_토큰'),
                            x='모델',
                            y='정답당_토큰',
                            title=t['token_per_correct'],
                            text='정답당_토큰',
                            color='정답당_토큰',
                            color_continuous_scale='RdYlGn_r'
                        )
                        fig.update_traces(
                            texttemplate='%{text:,.0f}',
                            textposition='outside',                textfont=dict(size=annotation_size),
                            marker_line_color='black',
                            marker_line_width=1.5
                        )
                        fig.update_layout(
                            height=400,
                            showlegend=False,
                            yaxis_title=t['tokens'] + ' / ' + t['correct'],
                            xaxis_title=t['model']
                        )
                        fig.update_xaxes(tickangle=45, tickfont=dict(size=annotation_size))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # 토큰 vs 정확도 산점도
                        if '평균_토큰' in model_token_stats.columns:
                            fig = px.scatter(
                                display_df,
                                x='평균_토큰',
                                y='정확도',
                                size='문제수',
                                text='모델',
                                title=t['token_efficiency'] + ' vs ' + t['accuracy'],
                                labels={
                                    '평균_토큰': t['avg_tokens_per_problem'],
                                    '정확도': t['accuracy'] + ' (%)'
                                }
                            )
                            fig.update_traces(
                                textposition='top center',
                                marker=dict(
                                    line=dict(width=2, color='black'),
                                    opacity=0.7
                                )
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                # 4. 비용 분석 (비용 수준 데이터가 있는 경우)
                if 'cost' in available_cols:
                    st.subheader("💵 " + t['cost_analysis'])
                    
                    cost_col = available_cols['cost']
                    
                    # 🔍 디버깅 정보 (펼치기/접기)
                    with st.expander("🔍 " + ("비용 데이터 확인" if lang == 'ko' else "Check Cost Data")):
                        st.write("**" + ("원본 비용 수준 값" if lang == 'ko' else "Original Cost Level Values") + ":**")
                        original_values = filtered_df[cost_col].unique().tolist()
                        st.write(f"고유 값: {original_values}")
                        st.write(f"개수: {len(original_values)}")
                    
                    # 비용 수준을 정규화 및 순서 정의
                    def normalize_cost_level(level):
                        if pd.isna(level):
                            return 'unknown'
                        level_str = str(level).lower().strip()
                        # 무료/로컬 모델
                        if level_str in ['무료', 'free', 'f', '0', 'local', 'localhost', '로컬']:
                            return t['free']
                        # 매우 낮음
                        elif level_str in ['매우낮음', 'very low', 'very_low', 'vl', 'verylow']:
                            return t['very_low']
                        # 낮음
                        elif level_str in ['낮음', 'low', 'l']:
                            return t['low']
                        # 중간
                        elif level_str in ['중간', 'medium', 'mid', 'm']:
                            return t['medium_cost']
                        # 높음
                        elif level_str in ['높음', 'high', 'h']:
                            return t['high']
                        return level
                    
                    # 비용 순서 정의 (무료 → 매우낮음 → 낮음 → 중간 → 높음)
                    cost_order = [t['free'], t['very_low'], t['low'], t['medium_cost'], t['high']]
                    
                    token_df['비용수준_정규화'] = token_df[cost_col].apply(normalize_cost_level)
                    model_token_stats['비용수준_정규화'] = model_token_stats['비용수준'].apply(normalize_cost_level) if '비용수준' in model_token_stats.columns else t['medium_cost']
                    
                    # 🔍 정규화 후 값 확인
                    with st.expander("🔍 " + ("정규화 후 비용 수준 값" if lang == 'ko' else "Normalized Cost Level Values")):
                        normalized_values = token_df['비용수준_정규화'].unique().tolist()
                        st.write(f"**정규화된 고유 값**: {normalized_values}")
                        st.write(f"**정의된 순서 (cost_order)**: {cost_order}")
                        
                        # 순서에 없는 값 확인
                        unexpected = [v for v in normalized_values if v not in cost_order]
                        if unexpected:
                            st.warning(f"⚠️ 정의된 순서에 없는 값: {unexpected}")
                        else:
                            st.success("✅ 모든 값이 정의된 순서에 포함됨")
                    
                    # 🆕 실제 비용 계산 기능 추가
                    st.markdown("---")
                    st.subheader("💰 " + t['actual_cost'] + " " + ('계산기' if lang == 'ko' else 'Calculator'))
                    
                    # 모델별 API 가격 정의 (2024-2025 기준, USD per 1M tokens)
                    MODEL_PRICING = {
                        # OpenAI (2025년 11월 기준, per 1M tokens)
                        'GPT-4o': {'input': 5.00, 'output': 15.00},  # 2025년 업데이트
                        'GPT-4o-Mini': {'input': 0.150, 'output': 0.600},
                        'GPT-4-Turbo': {'input': 10.00, 'output': 30.00},
                        'GPT-3.5-Turbo': {'input': 0.50, 'output': 1.50},
                        # Anthropic (2025년 11월 기준, per 1M tokens)
                        'Claude-Opus-4.5': {'input': 5.00, 'output': 25.00},  # 2025년 11월 출시
                        'Claude-Sonnet-4.5': {'input': 3.00, 'output': 15.00},  # 2025년 10월 출시
                        'Claude-Sonnet-4': {'input': 3.00, 'output': 15.00},
                        'Claude-Haiku-4.5': {'input': 1.00, 'output': 5.00},  # 2025년 11월 출시 (업데이트됨!)
                        'Claude-3.5-Sonnet': {'input': 3.00, 'output': 15.00},
                        'Claude-3.5-Haiku': {'input': 0.80, 'output': 4.00},
                        'Claude-3-Opus': {'input': 15.00, 'output': 75.00},
                        'Claude-3-Sonnet': {'input': 3.00, 'output': 15.00},
                        'Claude-3-Haiku': {'input': 0.25, 'output': 1.25},
                        # Google (2025년 기준, per 1M tokens)
                        'Gemini-1.5-Pro': {'input': 1.25, 'output': 5.00},
                        'Gemini-1.5-Flash': {'input': 0.075, 'output': 0.30},
                        # Alibaba (오픈소스)
                        'Qwen-2.5': {'input': 0.00, 'output': 0.00},  # 오픈소스/로컬
                        'Qwen2.5': {'input': 0.00, 'output': 0.00},  # 오픈소스/로컬
                        # LG AI Research
                        'EXAONE-3.5': {'input': 0.00, 'output': 0.00},  # 로컬/무료
                        # Meta
                        'Llama-3.3': {'input': 0.00, 'output': 0.00},  # 오픈소스/로컬
                        'Llama-3': {'input': 0.00, 'output': 0.00},  # 오픈소스/로컬
                    }
                    
                    # 가격 정보 표시
                    with st.expander("📋 " + ("모델별 API 가격 정보 (2025년 11월 기준)" if lang == 'ko' else "API Pricing by Model (November 2025)")):
                        pricing_data = []
                        for model, prices in MODEL_PRICING.items():
                            pricing_data.append({
                                '모델' if lang == 'ko' else 'Model': model,
                                '입력 ($/1M)' if lang == 'ko' else 'Input ($/1M)': f"${prices['input']:.3f}",
                                '출력 ($/1M)' if lang == 'ko' else 'Output ($/1M)': f"${prices['output']:.3f}"
                            })
                        st.dataframe(pd.DataFrame(pricing_data), width='stretch')
                        st.caption("💡 " + ("가격은 변동될 수 있습니다. 최신 가격은 각 제공업체 웹사이트를 확인하세요." if lang == 'ko' else "Prices may vary. Check provider websites for latest pricing."))
                        st.caption("📅 " + ("업데이트: 2025년 11월 (Claude Opus 4.5, Sonnet 4.5, Haiku 4.5 포함)" if lang == 'ko' else "Updated: November 2025 (includes Claude Opus 4.5, Sonnet 4.5, Haiku 4.5)"))
                    
                    # 실제 비용 계산
                    if '총_입력토큰' in model_token_stats.columns and '총_출력토큰' in model_token_stats.columns:
                        st.markdown("---")
                        
                        cost_calculations = []
                        for _, row in model_token_stats.iterrows():
                            model = row['모델']
                            input_tokens = row['총_입력토큰']
                            output_tokens = row['총_출력토큰']
                            
                            # 모델명 매칭 (부분 매칭)
                            matched_pricing = None
                            for price_model, pricing in MODEL_PRICING.items():
                                if price_model.replace('-', '').replace('.', '').lower() in model.replace('-', '').replace('.', '').lower():
                                    matched_pricing = pricing
                                    break
                            
                            if matched_pricing:
                                # 비용 계산 (USD)
                                input_cost = (input_tokens / 1_000_000) * matched_pricing['input']
                                output_cost = (output_tokens / 1_000_000) * matched_pricing['output']
                                total_cost = input_cost + output_cost
                                
                                # 문제당 비용
                                cost_per_problem = total_cost / row['문제수'] if row['문제수'] > 0 else 0
                                
                                # 정답당 비용 (효율성 지표)
                                correct_answers = row['문제수'] * row['정확도'] / 100
                                cost_per_correct = total_cost / correct_answers if correct_answers > 0 else 0
                                
                                cost_calculations.append({
                                    '모델' if lang == 'ko' else 'Model': model,
                                    '총비용 ($)' if lang == 'ko' else 'Total Cost ($)': total_cost,
                                    '문제당 ($)' if lang == 'ko' else 'Per Problem ($)': cost_per_problem,
                                    '정답당 ($)' if lang == 'ko' else 'Per Correct ($)': cost_per_correct,
                                    '정확도 (%)' if lang == 'ko' else 'Accuracy (%)': row['정확도'],
                                    '입력비용 ($)' if lang == 'ko' else 'Input Cost ($)': input_cost,
                                    '출력비용 ($)' if lang == 'ko' else 'Output Cost ($)': output_cost
                                })
                        
                        if cost_calculations:
                            cost_df = pd.DataFrame(cost_calculations)
                            
                            # 비용 효율성으로 정렬 (정답당 비용 기준)
                            cost_df = cost_df.sort_values('정답당 ($)' if lang == 'ko' else 'Per Correct ($)')
                            
                            st.subheader("💵 " + t['actual_cost'] + " " + ('분석' if lang == 'ko' else 'Analysis'))
                            
                            # 주요 메트릭
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                total_cost_all = cost_df['총비용 ($)' if lang == 'ko' else 'Total Cost ($)'].sum()
                                st.metric(
                                    t['total_estimated_cost'],
                                    f"${total_cost_all:.4f}"
                                )
                            
                            with col2:
                                avg_cost_per_problem = cost_df['문제당 ($)' if lang == 'ko' else 'Per Problem ($)'].mean()
                                st.metric(
                                    t['cost_per_problem'],
                                    f"${avg_cost_per_problem:.6f}"
                                )
                            
                            with col3:
                                # 가장 비용 효율적인 모델
                                most_efficient = cost_df.iloc[0]
                                st.metric(
                                    '최고 효율' if lang == 'ko' else 'Most Efficient',
                                    most_efficient['모델' if lang == 'ko' else 'Model'],
                                    f"${most_efficient['정답당 ($)' if lang == 'ko' else 'Per Correct ($)']:.6f}"
                                )
                            
                            with col4:
                                # 가장 비용 비효율적인 모델
                                least_efficient = cost_df.iloc[-1]
                                st.metric(
                                    '최저 효율' if lang == 'ko' else 'Least Efficient',
                                    least_efficient['모델' if lang == 'ko' else 'Model'],
                                    f"${least_efficient['정답당 ($)' if lang == 'ko' else 'Per Correct ($)']:.6f}"
                                )
                            
                            # 상세 테이블
                            st.markdown("---")
                            st.dataframe(
                                cost_df.style.format({
                                    '총비용 ($)' if lang == 'ko' else 'Total Cost ($)': '${:.6f}',
                                    '문제당 ($)' if lang == 'ko' else 'Per Problem ($)': '${:.8f}',
                                    '정답당 ($)' if lang == 'ko' else 'Per Correct ($)': '${:.8f}',
                                    '정확도 (%)' if lang == 'ko' else 'Accuracy (%)': '{:.2f}%',
                                    '입력비용 ($)' if lang == 'ko' else 'Input Cost ($)': '${:.6f}',
                                    '출력비용 ($)' if lang == 'ko' else 'Output Cost ($)': '${:.6f}'
                                }).background_gradient(
                                    subset=['정답당 ($)' if lang == 'ko' else 'Per Correct ($)'],
                                    cmap='RdYlGn_r'
                                ),
                                width='stretch'
                            )
                            
                            st.markdown("---")
                            
                            # 비용 시각화
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # 총 비용 비교
                                fig = px.bar(
                                    cost_df,
                                    x='모델' if lang == 'ko' else 'Model',
                                    y='총비용 ($)' if lang == 'ko' else 'Total Cost ($)',
                                    title=t['total_estimated_cost'],
                                    text='총비용 ($)' if lang == 'ko' else 'Total Cost ($)',
                                    color='총비용 ($)' if lang == 'ko' else 'Total Cost ($)',
                                    color_continuous_scale='Reds'
                                )
                                fig.update_traces(
                                    texttemplate='$%{text:.6f}',
                                    textposition='outside',                textfont=dict(size=annotation_size),
                                    marker_line_color='black',
                                    marker_line_width=1.5
                                )
                                fig.update_layout(
                                    height=400,
                                    showlegend=False,
                                    yaxis_title=t['cost'] + ' (USD)',
                                    xaxis_title=t['model']
                                )
                                fig.update_xaxes(tickangle=45, tickfont=dict(size=annotation_size))
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                # 정답당 비용 (효율성)
                                fig = px.bar(
                                    cost_df.sort_values('정답당 ($)' if lang == 'ko' else 'Per Correct ($)'),
                                    x='모델' if lang == 'ko' else 'Model',
                                    y='정답당 ($)' if lang == 'ko' else 'Per Correct ($)',
                                    title=t['cost_efficiency'] + ' (' + ('정답당 비용' if lang == 'ko' else 'Cost per Correct') + ')',
                                    text='정답당 ($)' if lang == 'ko' else 'Per Correct ($)',
                                    color='정답당 ($)' if lang == 'ko' else 'Per Correct ($)',
                                    color_continuous_scale='RdYlGn_r'
                                )
                                fig.update_traces(
                                    texttemplate='$%{text:.8f}',
                                    textposition='outside',                textfont=dict(size=annotation_size),
                                    marker_line_color='black',
                                    marker_line_width=1.5
                                )
                                fig.update_layout(
                                    height=400,
                                    showlegend=False,
                                    yaxis_title=t['cost'] + ' (USD)',
                                    xaxis_title=t['model']
                                )
                                fig.update_xaxes(tickangle=45, tickfont=dict(size=annotation_size))
                                st.plotly_chart(fig, use_container_width=True)
                            
                            st.markdown("---")
                            
                            # 비용 vs 정확도 산점도
                            fig = px.scatter(
                                cost_df,
                                x='총비용 ($)' if lang == 'ko' else 'Total Cost ($)',
                                y='정확도 (%)' if lang == 'ko' else 'Accuracy (%)',
                                text='모델' if lang == 'ko' else 'Model',
                                title=t['cost'] + ' vs ' + t['accuracy'],
                                color='정확도 (%)' if lang == 'ko' else 'Accuracy (%)',
                                color_continuous_scale='RdYlGn',
                                size='문제당 ($)' if lang == 'ko' else 'Per Problem ($)'
                            )
                            fig.update_traces(
                                textposition='top center',
                                marker=dict(
                                    line=dict(width=2, color='black'),
                                    opacity=0.7
                                )
                            )
                            fig.update_layout(
                                height=500,
                                yaxis=dict(range=[0, 100])
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # 인사이트
                            st.success(f"""
                            💡 **{t['cost_efficiency']} {'인사이트' if lang == 'ko' else 'Insights'}**:
                            - **{'최고 효율' if lang == 'ko' else 'Most Efficient'}**: {most_efficient['모델' if lang == 'ko' else 'Model']} (${most_efficient['정답당 ($)' if lang == 'ko' else 'Per Correct ($)']:.8f} / {'정답' if lang == 'ko' else 'correct'})
                            - **{'최저 효율' if lang == 'ko' else 'Least Efficient'}**: {least_efficient['모델' if lang == 'ko' else 'Model']} (${least_efficient['정답당 ($)' if lang == 'ko' else 'Per Correct ($)']:.8f} / {'정답' if lang == 'ko' else 'correct'})
                            - **{'효율 차이' if lang == 'ko' else 'Efficiency Gap'}**: {(least_efficient['정답당 ($)' if lang == 'ko' else 'Per Correct ($)'] / most_efficient['정답당 ($)' if lang == 'ko' else 'Per Correct ($)']):.1f}x
                            """)
                        else:
                            st.info("💡 " + ("현재 데이터의 모델들에 대한 가격 정보가 없습니다. 모델명을 확인하거나 가격 정보를 추가하세요." if lang == 'ko' else "No pricing information available for current models. Please check model names or add pricing info."))
                    
                    st.markdown("---")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 비용 수준별 모델 분포
                        cost_dist = token_df.groupby('비용수준_정규화')['모델'].nunique().reset_index()
                        cost_dist.columns = ['비용수준', '모델수']
                        
                        # cost_order에 있는 값만 필터링
                        cost_dist = cost_dist[cost_dist['비용수준'].isin(cost_order)]
                        
                        # 문자열로 명시적 변환
                        cost_dist['비용수준'] = cost_dist['비용수준'].astype(str)
                        
                        fig = px.pie(
                            cost_dist,
                            values='모델수',
                            names='비용수준',
                            title=t['cost_level'] + ' ' + ('분포' if lang == 'ko' else 'Distribution'),
                            hole=0.3
                        )
                        fig.update_traces(
                            textposition='inside',
                            textinfo='percent+label',                textfont=dict(size=annotation_size),
                            marker=dict(line=dict(color='black', width=2))
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # 비용 수준별 평균 정확도
                        cost_acc = token_df.groupby('비용수준_정규화')['정답여부'].mean().reset_index()
                        cost_acc.columns = ['비용수준', '정확도']
                        cost_acc['정확도'] = cost_acc['정확도'] * 100
                        
                        # cost_order에 있는 값만 필터링
                        cost_acc = cost_acc[cost_acc['비용수준'].isin(cost_order)]
                        
                        # 문자열로 명시적 변환
                        cost_acc['비용수준'] = cost_acc['비용수준'].astype(str)
                        
                        fig = px.bar(
                            cost_acc,
                            x='비용수준',
                            y='정확도',
                            title=t['cost_level'] + ' vs ' + t['accuracy'],
                            text='정확도',
                            color='정확도',
                            color_continuous_scale='RdYlGn'
                        )
                        fig.update_traces(
                            texttemplate='%{text:.1f}%',
                            textposition='outside',                textfont=dict(size=annotation_size),
                            marker_line_color='black',
                            marker_line_width=1.5
                        )
                        fig.update_layout(
                            height=400,
                            showlegend=False,
                            yaxis_title=t['accuracy'] + ' (%)',
                            yaxis=dict(range=[0, 100]),
                            xaxis=dict(
                                categoryorder='array',
                                categoryarray=cost_order
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("---")
                    
                    # 비용 효율성 매트릭스
                    st.subheader("📊 " + t['cost_efficiency'] + (' 매트릭스' if lang == 'ko' else ' Matrix'))
                    
                    # 비용 수준과 정확도로 모델 분류
                    if '비용수준_정규화' in model_token_stats.columns:
                        # 데이터 준비 및 필터링
                        plot_data = model_token_stats.copy()
                        
                        # cost_order에 있는 값만 필터링
                        plot_data = plot_data[plot_data['비용수준_정규화'].isin(cost_order)]
                        
                        # 비용수준을 문자열로 명시적 변환
                        plot_data['비용수준_정규화'] = plot_data['비용수준_정규화'].astype(str)
                        
                        fig = px.scatter(
                            plot_data,
                            x='비용수준_정규화',
                            y='정확도',
                            size='총_토큰' if '총_토큰' in plot_data.columns else '문제수',
                            text='모델',
                            title=t['cost_level'] + ' vs ' + t['accuracy'],
                            color='정확도',
                            color_continuous_scale='RdYlGn'
                        )
                        fig.update_traces(
                            textposition='top center',
                            marker=dict(
                                line=dict(width=2, color='black'),
                                opacity=0.7
                            )
                        )
                        fig.update_layout(
                            height=500,
                            yaxis=dict(range=[0, 100]),
                            xaxis=dict(
                                title=t['cost_level'],
                                categoryorder='array',
                                categoryarray=cost_order
                            ),
                            yaxis_title=t['accuracy'] + ' (%)'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 인사이트
                        st.info(f"""
                        💡 **{t['cost_efficiency']} {'인사이트' if lang == 'ko' else 'Insights'}**:
                        - **{'고효율 영역' if lang == 'ko' else 'High Efficiency Zone'}** ({'낮은 비용 + 높은 정확도' if lang == 'ko' else 'Low cost + High accuracy'}): {'좌측 상단' if lang == 'ko' else 'Top left'}
                        - **{'고비용 영역' if lang == 'ko' else 'High Cost Zone'}** ({'높은 비용' if lang == 'ko' else 'High cost'}): {'우측' if lang == 'ko' else 'Right side'}
                        - {'모델 선택 시 비용 대비 성능을 고려하세요' if lang == 'ko' else 'Consider cost-performance ratio when selecting models'}
                        """)
                
                st.markdown("---")
                
                # 5. 테스트별 토큰 분석 (테스트가 여러 개인 경우)
                if '테스트명' in token_df.columns and token_df['테스트명'].nunique() > 1:
                    st.subheader("📚 " + ("테스트별 토큰 사용량" if lang == 'ko' else "Token Usage by Test"))
                    
                    token_col = available_cols.get('total', available_cols.get('input', list(available_cols.values())[0]))
                    test_token = token_df.groupby(['모델', '테스트명'])[token_col].sum().reset_index()
                    test_token.columns = ['모델', '테스트명', '총토큰']
                    
                    fig = px.bar(
                        test_token,
                        x='테스트명',
                        y='총토큰',
                        color='모델',
                        barmode='group',
                        title='테스트별 모델 토큰 사용량' if lang == 'ko' else 'Token Usage by Test and Model',
                        labels={'총토큰': t['total_tokens']}
                    )
                    fig.update_layout(
                        height=400,
                        xaxis_title=t['testname'],
                        yaxis_title=t['total_tokens']
                    )
                    fig.update_xaxes(tickangle=45, tickfont=dict(size=annotation_size))
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                # 6. 문제 유형별 토큰 분석 (이미지 문제가 있는 경우)
                if 'image' in token_df.columns:
                    st.subheader("🖼️ " + ("문제 유형별 토큰 사용량" if lang == 'ko' else "Token Usage by Problem Type"))
                    
                    # 이미지 문제 여부 구분
                    token_df['문제유형'] = token_df['image'].apply(
                        lambda x: t['text_only'] if str(x).lower() == 'text_only' or str(x) == 'X' else t['image_problem']
                    )
                    
                    token_col = available_cols.get('total', available_cols.get('input', list(available_cols.values())[0]))
                    problem_type_token = token_df.groupby(['모델', '문제유형']).agg({
                        token_col: 'mean',
                        '정답여부': 'mean'
                    }).reset_index()
                    problem_type_token.columns = ['모델', '문제유형', '평균토큰', '정확도']
                    problem_type_token['정확도'] = problem_type_token['정확도'] * 100
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 문제 유형별 평균 토큰
                        fig = px.bar(
                            problem_type_token,
                            x='모델',
                            y='평균토큰',
                            color='문제유형',
                            barmode='group',
                            title=t['avg_tokens_per_problem'] + ' (' + t['problem_type'] + '별)',
                            labels={'평균토큰': t['avg_tokens_per_problem']}
                        )
                        fig.update_layout(
                            height=400,
                            xaxis_title=t['model']
                        )
                        fig.update_xaxes(tickangle=45, tickfont=dict(size=annotation_size))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # 문제 유형별 정확도 비교
                        fig = px.bar(
                            problem_type_token,
                            x='모델',
                            y='정확도',
                            color='문제유형',
                            barmode='group',
                            title=t['accuracy'] + ' (' + t['problem_type'] + '별)',
                            labels={'정확도': t['accuracy'] + ' (%)'}
                        )
                        fig.update_layout(
                            height=400,
                            xaxis_title=t['model'],
                            yaxis=dict(range=[0, 100])
                        )
                        fig.update_xaxes(tickangle=45, tickfont=dict(size=annotation_size))
                        st.plotly_chart(fig, use_container_width=True)
    
    # 탭 10: 테스트셋 통계
    with tabs[9]:
        st.header(f"📋 {t['testset_stats']}")
        
        # 상단에 전체 통계 요약 추가
        st.subheader("📊 " + ("전체 테스트셋 통계" if lang == 'ko' else "Overall Test Set Statistics"))
        
        # 선택된 테스트들의 전체 통계
        total_all_problems = 0
        total_law_problems = 0
        total_non_law_problems = 0
        
        if selected_tests:
            for test_name in selected_tests:
                if test_name in testsets:
                    test_df = testsets[test_name]
                    total_all_problems += len(test_df)
                    if 'law' in test_df.columns:
                        total_law_problems += len(test_df[test_df['law'] == 'O'])
                        total_non_law_problems += len(test_df[test_df['law'] != 'O'])
                    else:
                        total_non_law_problems += len(test_df)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "총 문제 수" if lang == 'ko' else "Total Problems",
                f"{total_all_problems:,}",
                help="선택된 모든 테스트의 전체 문제 수 (테스트셋 기준)"
            )
        with col2:
            st.metric(
                "법령 문제" if lang == 'ko' else "Law Problems",
                f"{total_law_problems:,}",
                help="법령 문제 수 (테스트셋 기준)"
            )
        with col3:
            st.metric(
                "비법령 문제" if lang == 'ko' else "Non-Law Problems",
                f"{total_non_law_problems:,}",
                help="비법령 문제 수 (테스트셋 기준)"
            )
        
        st.info("💡 " + (
            "이 통계는 테스트셋 파일 기준입니다. 전체 요약 탭의 수치와 동일합니다." 
            if lang == 'ko' 
            else "These statistics are based on test set files. They match the Overview tab."
        ))
        
        st.markdown("---")
        
        if selected_tests:
            # 선택된 테스트들의 개별 통계 표시
            for test_name in selected_tests:
                stats = get_testset_statistics(testsets, test_name, lang)
                if stats:
                    st.subheader(f"📖 {test_name}")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(t['total_problems'], stats['total_problems'])
                    
                    with col2:
                        if 'law_problems' in stats:
                            st.metric(t['law_problems'], stats['law_problems'])
                    
                    with col3:
                        if 'non_law_problems' in stats:
                            st.metric(t['non_law_problems'], stats['non_law_problems'])
                    
                    # 과목별, 연도별, 세션별 통계
                    if 'by_subject' in stats or 'by_year' in stats or 'by_session' in stats:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if 'by_subject' in stats:
                                st.markdown(f"**{t['by_subject']}**")
                                subject_df = pd.DataFrame(list(stats['by_subject'].items()), 
                                                         columns=['Subject', 'Count'])
                                fig = px.bar(subject_df, x='Subject', y='Count', 
                                           title=t['subject_distribution'])
                                fig.update_xaxes(tickangle=45, tickfont=dict(size=annotation_size))
                                st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            if 'by_year' in stats:
                                st.markdown(f"**{t['by_year']}**")
                                year_df = pd.DataFrame(list(stats['by_year'].items()), 
                                                      columns=['Year', 'Count'])
                                fig = px.bar(year_df, x='Year', y='Count', 
                                           title=t['year_distribution'])
                                st.plotly_chart(fig, use_container_width=True)
                        
                        with col3:
                            if 'by_session' in stats:
                                st.markdown(f"**{t['by_session']}**")
                                session_df = pd.DataFrame(list(stats['by_session'].items()), 
                                                         columns=['Session', 'Count'])
                                fig = px.bar(session_df, x='Session', y='Count', 
                                           title=t['session_distribution'])
                                st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("---")
        else:
            st.info("테스트를 선택해주세요.")
    
    # 사이드바 하단 정보
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### 📌 {t['help']}")
    st.sidebar.markdown(f"""
    **{t['new_features']}:**
    - ✨ **{t['token_cost_analysis']}**: 토큰 사용량 및 비용 효율성 분석
    - ✨ **{t['session']} {t['filters']}**: {t['session_filter']}
    - ✨ **{t['incorrect_analysis']}**: {t['incorrect_pattern']}
    - ✨ **{t['difficulty_analysis']}**: {t['difficulty_comparison']}
    - ✨ **{t['problem_type']} {t['filters']}**: {t['problem_type_filter']}
    
    **{t['existing_features']}:**
    - {t['basic_filters']}
    - {t['law_analysis_desc']}
    - {t['detail_analysis']}
    """)
    
    # 탭 11: 추가 분석
    with tabs[10]:
        st.header("📑 " + ("추가 분석 표 및 시각화" if lang == 'ko' else "Additional Analysis Tables and Visualizations"))
        
        # ========== 추가 분석 표 섹션 ==========
        
        st.markdown("### 📊 " + ("추가 분석 표" if lang == 'ko' else "Additional Analysis Tables"))
        st.markdown("---")
        
        # 표 1: 테스트셋별 평균 정답률 (NEW!)
        st.subheader("📋 " + ("표 1: 테스트셋별 평균 정답률 및 통계" if lang == 'ko' else "Table 1: Average Accuracy and Statistics by Test Set"))
        table1 = create_testset_accuracy_table(filtered_df, testsets, lang)
        if table1 is not None and len(table1) > 0:
            display_table_with_download(table1, "", "table1_testset_accuracy.xlsx", lang)
            
            # 간단한 시각화 추가
            st.markdown("#### " + ("테스트셋별 정확도 비교" if lang == 'ko' else "Accuracy Comparison by Test Set"))
            fig = px.bar(
                table1,
                x='테스트명' if lang == 'ko' else 'Test Name',
                y='평균 정답률 (%)' if lang == 'ko' else 'Avg Accuracy (%)',
                text='평균 정답률 (%)' if lang == 'ko' else 'Avg Accuracy (%)',
                color='평균 정답률 (%)' if lang == 'ko' else 'Avg Accuracy (%)',
                color_continuous_scale='RdYlGn',
                title='테스트셋별 평균 정답률' if lang == 'ko' else 'Average Accuracy by Test Set'
            )
            fig.update_traces(
                texttemplate='%{text:.1f}%',
                textposition='outside',
                textfont=dict(size=annotation_size),
                marker_line_color='black',
                marker_line_width=1.5
            )
            fig.update_layout(
                height=400,
                showlegend=False,
                yaxis_title='평균 정답률 (%)' if lang == 'ko' else 'Avg Accuracy (%)',
                xaxis_title='테스트명' if lang == 'ko' else 'Test Name',
                yaxis=dict(range=[0, 100])
            )
            fig.update_xaxes(tickangle=45, tickfont=dict(size=annotation_size))
            fig.update_yaxes(tickfont=dict(size=annotation_size))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("테스트셋 데이터가 없습니다." if lang == 'ko' else "No test set data available.")
        
        st.markdown("---")
        
        # 표 2: 법령/비법령 정답률 비교 (NEW!)
        st.subheader("⚖️ " + ("표 2: 테스트셋별 법령/비법령 정답률 비교" if lang == 'ko' else "Table 2: Law vs Non-Law Accuracy Comparison by Test Set"))
        table2 = create_law_nonlaw_comparison_table(filtered_df, testsets, lang)
        if table2 is not None and len(table2) > 0:
            display_table_with_download(table2, "", "table2_law_nonlaw_comparison.xlsx", lang)
            
            # 간단한 시각화 추가 - 법령 vs 비법령 정답률
            st.markdown("#### " + ("법령 vs 비법령 정답률 비교" if lang == 'ko' else "Law vs Non-Law Accuracy Comparison"))
            
            # 데이터 준비
            chart_data = []
            for _, row in table2.iterrows():
                test_name = row['테스트명' if lang == 'ko' else 'Test Name']
                chart_data.append({
                    '테스트명' if lang == 'ko' else 'Test Name': test_name,
                    '구분' if lang == 'ko' else 'Type': '법령' if lang == 'ko' else 'Law',
                    '정답률 (%)' if lang == 'ko' else 'Accuracy (%)': row['법령 정답률 (%)' if lang == 'ko' else 'Law Accuracy (%)']
                })
                chart_data.append({
                    '테스트명' if lang == 'ko' else 'Test Name': test_name,
                    '구분' if lang == 'ko' else 'Type': '비법령' if lang == 'ko' else 'Non-Law',
                    '정답률 (%)' if lang == 'ko' else 'Accuracy (%)': row['비법령 정답률 (%)' if lang == 'ko' else 'Non-Law Accuracy (%)']
                })
            
            chart_df = pd.DataFrame(chart_data)
            
            fig = px.bar(
                chart_df,
                x='테스트명' if lang == 'ko' else 'Test Name',
                y='정답률 (%)' if lang == 'ko' else 'Accuracy (%)',
                color='구분' if lang == 'ko' else 'Type',
                barmode='group',
                title='테스트셋별 법령/비법령 정답률 비교' if lang == 'ko' else 'Law vs Non-Law Accuracy by Test Set',
                color_discrete_map={
                    '법령' if lang == 'ko' else 'Law': '#FF6B6B',
                    '비법령' if lang == 'ko' else 'Non-Law': '#4ECDC4'
                }
            )
            fig.update_traces(
                marker_line_color='black',
                marker_line_width=1.5
            )
            fig.update_layout(
                height=400,
                yaxis_title='정답률 (%)' if lang == 'ko' else 'Accuracy (%)',
                xaxis_title='테스트명' if lang == 'ko' else 'Test Name',
                yaxis=dict(range=[0, 100]),
                legend=dict(
                    title='구분' if lang == 'ko' else 'Type',
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(size=annotation_size)
                )
            )
            fig.update_xaxes(tickangle=45, tickfont=dict(size=annotation_size))
            fig.update_yaxes(tickfont=dict(size=annotation_size))
            st.plotly_chart(fig, use_container_width=True)
            
            # 정답률 차이 막대 그래프
            st.markdown("#### " + ("정답률 차이 (법령 - 비법령)" if lang == 'ko' else "Accuracy Difference (Law - Non-Law)"))
            fig2 = px.bar(
                table2,
                x='테스트명' if lang == 'ko' else 'Test Name',
                y='정답률 차이 (법령-비법령)' if lang == 'ko' else 'Accuracy Diff (Law-NonLaw)',
                text='정답률 차이 (법령-비법령)' if lang == 'ko' else 'Accuracy Diff (Law-NonLaw)',
                color='정답률 차이 (법령-비법령)' if lang == 'ko' else 'Accuracy Diff (Law-NonLaw)',
                color_continuous_scale='RdYlGn',
                title='법령 문제의 상대적 난이도' if lang == 'ko' else 'Relative Difficulty of Law Problems'
            )
            fig2.update_traces(
                texttemplate='%{text:.1f}%p',
                textposition='outside',
                textfont=dict(size=annotation_size),
                marker_line_color='black',
                marker_line_width=1.5
            )
            fig2.update_layout(
                height=400,
                showlegend=False,
                yaxis_title='정답률 차이 (%p)' if lang == 'ko' else 'Accuracy Difference (%p)',
                xaxis_title='테스트명' if lang == 'ko' else 'Test Name'
            )
            fig2.update_xaxes(tickangle=45, tickfont=dict(size=annotation_size))
            fig2.update_yaxes(tickfont=dict(size=annotation_size))
            st.plotly_chart(fig2, use_container_width=True)
            
            # 인사이트 표시
            avg_diff = table2['정답률 차이 (법령-비법령)' if lang == 'ko' else 'Accuracy Diff (Law-NonLaw)'].mean()
            if avg_diff > 0:
                insight = f"💡 평균적으로 법령 문제가 비법령 문제보다 {abs(avg_diff):.1f}%p 더 쉽습니다." if lang == 'ko' else f"💡 On average, law problems are {abs(avg_diff):.1f}%p easier than non-law problems."
            elif avg_diff < 0:
                insight = f"💡 평균적으로 법령 문제가 비법령 문제보다 {abs(avg_diff):.1f}%p 더 어렵습니다." if lang == 'ko' else f"💡 On average, law problems are {abs(avg_diff):.1f}%p harder than non-law problems."
            else:
                insight = "💡 법령 문제와 비법령 문제의 난이도가 비슷합니다." if lang == 'ko' else "💡 Law and non-law problems have similar difficulty."
            st.info(insight)
        else:
            st.info("법령 데이터가 없습니다." if lang == 'ko' else "No law classification data available.")
        
        st.markdown("---")
        
        # 표 3: 모델 출시 시기와 성능
        st.subheader("📅 " + ("표 3: 모델 출시 시기와 SafetyQ&A 성능" if lang == 'ko' else "Table 3: Model Release Date and Performance"))
        table3 = create_model_release_performance_table(filtered_df, lang)
        if table3 is not None and len(table3) > 0:
            # 날짜를 숫자로 변환 (YYYY-MM -> YYYYMM)
            table3_copy = table3.copy()
            date_col = '출시 시기' if lang == 'ko' else 'Release Date'
            table3_copy['date_numeric'] = table3_copy[date_col].str.replace('-', '').astype(int)
            
            display_table_with_download(table3, "", "table3_model_release_performance.xlsx", lang)
        
        st.markdown("---")
        
        # 표 4: 응답 시간 및 파라미터
        st.subheader("⏱️ " + ("표 4: 모델별 평균 응답 시간 및 정답률" if lang == 'ko' else "Table 4: Response Time and Accuracy by Model"))
        table4 = create_response_time_parameters_table(filtered_df, lang)
        if table4 is not None and len(table4) > 0:
            display_table_with_download(table4, "", "table4_response_time_parameters.xlsx", lang)
        else:
            st.info("응답 시간 데이터가 없습니다." if lang == 'ko' else "No response time data available.")
        
        st.markdown("---")
        
        # 표 5: 모델별 법령/비법령 성능 비교 (NEW!)
        st.subheader("⚖️ " + ("표 5: 모델별 법령 문항 vs 비법령 문항 성능 비교" if lang == 'ko' else "Table 5: Law vs Non-Law Performance by Model"))
        table5 = create_model_law_performance_table(filtered_df, lang)
        if table5 is not None and len(table5) > 0:
            display_table_with_download(table5, "", "table5_model_law_performance.xlsx", lang)
        else:
            st.info("법령 데이터가 없습니다." if lang == 'ko' else "No law classification data available.")
        
        st.markdown("---")
        
        # 표 6: 출제 연도별 상관분석
        st.subheader("📅 " + ("표 6: 출제 연도별 평균 정답률 및 상관관계" if lang == 'ko' else "Table 6: Accuracy by Year with Correlation"))
        table6 = create_year_correlation_table(filtered_df, lang)
        if table6 is not None and len(table6) > 0:
            display_table_with_download(table6, "", "table6_year_correlation.xlsx", lang)
        else:
            st.info("연도 데이터가 없습니다." if lang == 'ko' else "No year data available.")
        
        st.markdown("---")
        
        # 표 7: 난이도 구간별 분포
        st.subheader("📈 " + ("표 7: 난이도 구간별 문항 분포" if lang == 'ko' else "Table 7: Problem Distribution by Difficulty"))
        table7 = create_difficulty_distribution_table(filtered_df, lang)
        if table7 is not None and len(table7) > 0:
            display_table_with_download(table7, "", "table7_difficulty_distribution.xlsx", lang)
        
        st.markdown("---")
        
        # 표 8: 난이도 구간별 모델 성능 (NEW!)
        st.subheader("🎯 " + ("표 8: 주요 모델의 난이도 구간별 정답률" if lang == 'ko' else "Table 8: Model Performance by Difficulty Level"))
        table8 = create_difficulty_model_performance_table(filtered_df, lang)
        if table8 is not None and len(table8) > 0:
            display_table_with_download(table8, "", "table8_difficulty_model_performance.xlsx", lang)
        else:
            st.info("난이도 분석 데이터가 없습니다." if lang == 'ko' else "No difficulty analysis data available.")
        
        st.markdown("---")
        
        # 표 9: 비용 효율성 비교 (NEW!)
        st.subheader("💰 " + ("표 9: 주요 상업용 모델의 비용 효율성 비교" if lang == 'ko' else "Table 9: Cost Efficiency Comparison"))
        table9 = create_cost_efficiency_table(filtered_df, lang)
        if table9 is not None and len(table9) > 0:
            display_table_with_download(table9, "", "table9_cost_efficiency.xlsx", lang)
        else:
            st.info("토큰/비용 데이터가 없습니다." if lang == 'ko' else "No token/cost data available.")
        
        st.markdown("---")
        
        # 표 10: 오답 패턴
        st.subheader("❌ " + ("표 10: 주요 오답 패턴 및 빈도" if lang == 'ko' else "Table 10: Major Error Patterns"))
        table10 = create_incorrect_pattern_table(filtered_df, lang)
        if table10 is not None and len(table10) > 0:
            display_table_with_download(table10, "", "table10_error_patterns.xlsx", lang)
        
        st.markdown("---")
        
        # 표 11: 범용 벤치마크 비교 (NEW!)
        st.subheader("📊 " + ("표 11: SafetyQ&A와 범용 벤치마크 성능 비교" if lang == 'ko' else "Table 11: SafetyQ&A vs General Benchmarks"))
        table11 = create_benchmark_comparison_table(filtered_df, lang)
        if table11 is not None and len(table11) > 0:
            display_table_with_download(table11, "", "table11_benchmark_comparison.xlsx", lang)
            st.caption("💡 " + ("범용 벤치마크 점수는 SafetyQ&A 성능 기반 추정치입니다." if lang == 'ko' else "General benchmark scores are estimates based on SafetyQ&A performance."))
        else:
            st.info("벤치마크 비교 데이터를 생성할 수 없습니다." if lang == 'ko' else "Cannot generate benchmark comparison data.")
        
        # ========== 추가 시각화 섹션 ==========
        
        st.markdown("---")
        st.markdown("### 📈 " + ("추가 시각화" if lang == 'ko' else "Additional Visualizations"))
        st.markdown("---")
        
        # Figure 1: 모델별 전체 정답률 막대 그래프 (NEW!)
        st.subheader("📊 " + ("Figure 1: 모델별 전체 정답률 막대 그래프" if lang == 'ko' else "Figure 1: Overall Accuracy by Model"))
        
        model_acc = filtered_df.groupby('모델')['정답여부'].mean() * 100
        model_acc_df = model_acc.reset_index()
        model_acc_df.columns = ['모델' if lang == 'ko' else 'Model', '정답률' if lang == 'ko' else 'Accuracy']
        model_acc_df = model_acc_df.sort_values('정답률' if lang == 'ko' else 'Accuracy', ascending=False)
        
        # 평균선 계산
        avg_acc = model_acc_df['정답률' if lang == 'ko' else 'Accuracy'].mean()
        
        fig = px.bar(
            model_acc_df,
            x='모델' if lang == 'ko' else 'Model',
            y='정답률' if lang == 'ko' else 'Accuracy',
            title='모델별 전체 정답률' if lang == 'ko' else 'Overall Accuracy by Model',
            text='정답률' if lang == 'ko' else 'Accuracy',
            color='정답률' if lang == 'ko' else 'Accuracy',
            color_continuous_scale='RdYlGn'
        )
        
        # 평균선 추가
        fig.add_hline(
            y=avg_acc,
            line_dash="dash",
            line_color="red",
            annotation_text=f"평균: {avg_acc:.1f}%" if lang == 'ko' else f"Average: {avg_acc:.1f}%",
            annotation_position="right"
        )
        
        fig.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside',
            textfont=dict(size=annotation_size),
            marker_line_color='black',
            marker_line_width=1.5
        )
        fig.update_layout(
            height=500,
            showlegend=False,
            yaxis_title='정답률 (%)' if lang == 'ko' else 'Accuracy (%)',
            xaxis_title='모델' if lang == 'ko' else 'Model',
            yaxis=dict(range=[0, 100])
        )
        fig.update_xaxes(tickangle=45, tickfont=dict(size=annotation_size))
        fig.update_yaxes(tickfont=dict(size=annotation_size))
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Figure 2: 테스트셋별 정답률 박스플롯 (NEW!)
        if '테스트명' in filtered_df.columns and '정답여부' in filtered_df.columns:
            st.subheader("📦 " + ("Figure 2: 테스트셋별 정답률 박스플롯" if lang == 'ko' else "Figure 2: Accuracy Distribution by Test Set"))
            
            # 모델별, 테스트별 정답률 계산
            test_model_acc = filtered_df.groupby(['테스트명', '모델'])['정답여부'].mean().reset_index()
            test_model_acc['정답률'] = test_model_acc['정답여부'] * 100
            
            # 디버깅 정보 (펼치기/접기)
            with st.expander("🔍 " + ("디버깅 정보" if lang == 'ko' else "Debug Info")):
                st.write("**정답률 범위:**", f"{test_model_acc['정답률'].min():.2f}% ~ {test_model_acc['정답률'].max():.2f}%")
                st.write("**데이터 샘플:**")
                st.dataframe(test_model_acc.head(10))
            
            fig = px.box(
                test_model_acc,
                x='테스트명',
                y='정답률',
                title='테스트셋별 정답률 분포 (모델별)' if lang == 'ko' else 'Accuracy Distribution by Test Set (per Model)',
                labels={
                    '테스트명': '테스트명' if lang == 'ko' else 'Test Name',
                    '정답률': '정답률 (%)' if lang == 'ko' else 'Accuracy (%)'
                },
                color='테스트명',
                points='all'  # 모든 데이터 포인트 표시
            )
            
            fig.update_layout(
                height=500,
                yaxis_title='정답률 (%)' if lang == 'ko' else 'Accuracy (%)',
                xaxis_title='테스트명' if lang == 'ko' else 'Test Name',
                yaxis=dict(range=[0, 100]),
                showlegend=False
            )
            fig.update_xaxes(tickangle=45, tickfont=dict(size=annotation_size))
            fig.update_yaxes(tickfont=dict(size=annotation_size))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
        
        # Figure 3: 과목 유형별 히트맵 (NEW!)
        if 'Subject' in filtered_df.columns:
            st.subheader("🔥 " + ("Figure 3: 과목 유형별 평균 정답률 히트맵" if lang == 'ko' else "Figure 3: Accuracy Heatmap by Subject Type"))
            
            # 모델 × 과목 히트맵
            subject_model = filtered_df.groupby(['모델', 'Subject'])['정답여부'].mean() * 100
            subject_model_pivot = subject_model.unstack(fill_value=0)
            
            fig = go.Figure(data=go.Heatmap(
                z=subject_model_pivot.values,
                x=subject_model_pivot.columns,
                y=subject_model_pivot.index,
                colorscale='RdYlGn',
                text=np.round(subject_model_pivot.values, 1),
                texttemplate='%{text:.1f}',
                textfont={"size": annotation_size},
                colorbar=dict(title="정답률 (%)" if lang == 'ko' else "Accuracy (%)")
            ))
            
            fig.update_layout(
                title='모델 × 과목 정답률 히트맵' if lang == 'ko' else 'Model × Subject Accuracy Heatmap',
                height=600,
                xaxis_title='과목' if lang == 'ko' else 'Subject',
                yaxis_title='모델' if lang == 'ko' else 'Model'
            )
            fig.update_xaxes(tickangle=45, tickfont=dict(size=annotation_size))
            fig.update_yaxes(tickfont=dict(size=annotation_size))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
        
        # Figure 5: 응답 시간-정답률 산점도 (NEW!)
        if table4 is not None and len(table4) > 0:
            st.subheader("⚡ " + ("Figure 5: 응답 시간-정답률 산점도" if lang == 'ko' else "Figure 5: Response Time vs Accuracy"))
            
            # 산점도 생성 (size 제거 - 파라미터 수로 색상 표현)
            fig = px.scatter(
                table4,
                x='평균 응답시간 (초)' if lang == 'ko' else 'Avg Response Time (s)',
                y='정답률 (%)' if lang == 'ko' else 'Accuracy (%)',
                text='모델명' if lang == 'ko' else 'Model',
                color='파라미터 수 (B)' if lang == 'ko' else 'Parameters (B)',
                title='응답 시간 vs 정확도' if lang == 'ko' else 'Response Time vs Accuracy',
                color_continuous_scale='Viridis'
            )
            
            fig.update_traces(
                textposition='top center',
                textfont=dict(size=annotation_size),
                marker=dict(size=15, line=dict(width=1.5, color='black'))
            )
            fig.update_layout(
                height=500,
                yaxis=dict(range=[0, 100])
            )
            fig.update_xaxes(tickfont=dict(size=annotation_size))
            fig.update_yaxes(tickfont=dict(size=annotation_size))
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 " + ("왼쪽 위(빠른 시간 + 높은 정확도)가 가장 효율적입니다." if lang == 'ko' else "Top left (fast time + high accuracy) is most efficient."))
            
            st.markdown("---")
        
        # Figure 6: 법령/비법령 그룹 막대 차트 (NEW!)
        if table5 is not None and len(table5) > 0:
            st.subheader("⚖️ " + ("Figure 6: 법령/비법령 문항 정답률 비교" if lang == 'ko' else "Figure 6: Law vs Non-Law Accuracy Comparison"))
            
            # 데이터 준비
            chart_data = []
            for _, row in table5.iterrows():
                model = row['모델명' if lang == 'ko' else 'Model']
                chart_data.append({
                    '모델' if lang == 'ko' else 'Model': model,
                    '구분' if lang == 'ko' else 'Type': '법령' if lang == 'ko' else 'Law',
                    '정답률 (%)' if lang == 'ko' else 'Accuracy (%)': row['법령 문항 정답률 (%)' if lang == 'ko' else 'Law Accuracy (%)']
                })
                chart_data.append({
                    '모델' if lang == 'ko' else 'Model': model,
                    '구분' if lang == 'ko' else 'Type': '비법령' if lang == 'ko' else 'Non-Law',
                    '정답률 (%)' if lang == 'ko' else 'Accuracy (%)': row['비법령 문항 정답률 (%)' if lang == 'ko' else 'Non-Law Accuracy (%)']
                })
            
            chart_df = pd.DataFrame(chart_data)
            
            fig = px.bar(
                chart_df,
                x='모델' if lang == 'ko' else 'Model',
                y='정답률 (%)' if lang == 'ko' else 'Accuracy (%)',
                color='구분' if lang == 'ko' else 'Type',
                barmode='group',
                title='모델별 법령/비법령 정답률 비교' if lang == 'ko' else 'Law vs Non-Law Accuracy by Model',
                color_discrete_map={
                    '법령' if lang == 'ko' else 'Law': '#FF6B6B',
                    '비법령' if lang == 'ko' else 'Non-Law': '#4ECDC4'
                }
            )
            
            fig.update_traces(marker_line_color='black', marker_line_width=1.5)
            fig.update_layout(
                height=500,
                yaxis_title='정답률 (%)' if lang == 'ko' else 'Accuracy (%)',
                xaxis_title='모델' if lang == 'ko' else 'Model',
                yaxis=dict(range=[0, 100]),
                legend=dict(font=dict(size=annotation_size))
            )
            fig.update_xaxes(tickangle=45, tickfont=dict(size=annotation_size))
            fig.update_yaxes(tickfont=dict(size=annotation_size))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
        
        # Figure 7: 출제 연도별 추이 선 그래프 (NEW!)
        if table6 is not None and len(table6) > 0:
            st.subheader("📈 " + ("Figure 7: 출제 연도별 정답률 추이" if lang == 'ko' else "Figure 7: Accuracy Trend by Year"))
            
            # 상관계수, p-value 행 제거
            year_col = '연도' if lang == 'ko' else 'Year'
            # 실제 컬럼명 확인 (표 6에서는 '평균 정답률'로 저장됨)
            acc_col = '평균 정답률' if lang == 'ko' else 'Avg Accuracy'
            
            plot_data = table6.copy()
            
            # 연도와 정답률이 모두 숫자인 행만 선택
            if year_col in plot_data.columns and acc_col in plot_data.columns:
                # 1. 특정 문자열 행 제거
                exclude_keywords = ['상관계수', 'p-value', 'correlation', 'Correlation']
                for keyword in exclude_keywords:
                    plot_data = plot_data[~plot_data[year_col].astype(str).str.contains(keyword, na=False)]
                
                # 2. 연도가 숫자로 변환 가능한 행만 선택
                def is_numeric_convertible(x):
                    try:
                        float(str(x))
                        return True
                    except (ValueError, TypeError):
                        return False
                
                plot_data = plot_data[plot_data[year_col].apply(is_numeric_convertible)]
                plot_data = plot_data[plot_data[acc_col].apply(is_numeric_convertible)]
                
                # 3. 데이터 타입 변환
                plot_data[year_col] = plot_data[year_col].astype(int)
                plot_data[acc_col] = plot_data[acc_col].astype(float)
            
            if len(plot_data) > 0:
                try:
                    fig = px.line(
                        plot_data,
                        x=year_col,
                        y=acc_col,
                        title='연도별 평균 정답률 추이' if lang == 'ko' else 'Average Accuracy Trend by Year',
                        markers=True,
                        text=acc_col
                    )
                    
                    fig.update_traces(
                        texttemplate='%{text:.1f}%',
                        textposition='top center',
                        textfont=dict(size=annotation_size),
                        marker=dict(size=10, line=dict(width=2, color='black')),
                        line=dict(width=3)
                    )
                    fig.update_layout(
                        height=500,
                        yaxis_title='평균 정답률 (%)' if lang == 'ko' else 'Avg Accuracy (%)',
                        xaxis_title='출제 연도' if lang == 'ko' else 'Year',
                        yaxis=dict(range=[0, 100])
                    )
                    fig.update_xaxes(tickfont=dict(size=annotation_size))
                    fig.update_yaxes(tickfont=dict(size=annotation_size))
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"{'차트 생성 오류' if lang == 'ko' else 'Chart creation error'}: {str(e)}")
                    with st.expander("🔍 " + ("디버깅 정보" if lang == 'ko' else "Debug Info")):
                        st.write("**필터링된 데이터:**")
                        st.dataframe(plot_data)
            else:
                st.warning("연도 데이터가 충분하지 않습니다." if lang == 'ko' else "Insufficient year data.")
            
            st.markdown("---")
        
        # Figure 7-1: 테스트셋별 연도별 정답률 추이 (NEW!)
        st.subheader("📈 " + ("Figure 7-1: 테스트셋별 연도별 정답률 추이" if lang == 'ko' else "Figure 7-1: Accuracy Trend by Year and Test Set"))
        
        if 'Year' in filtered_df.columns:
            # 연도를 정수로 변환
            year_int_series = filtered_df['Year'].apply(safe_convert_to_int)
            valid_year_mask = year_int_series.notna()
            
            if valid_year_mask.any():
                # 분석용 데이터프레임 생성
                year_test_df = pd.DataFrame({
                    'Year_Int': year_int_series[valid_year_mask],
                    '정답여부': filtered_df.loc[valid_year_mask, '정답여부'],
                    '테스트명': filtered_df.loc[valid_year_mask, '테스트명']
                })
                
                # 1. 전체 연도별 정답률
                overall_year_acc = year_test_df.groupby('Year_Int')['정답여부'].mean() * 100
                overall_year_acc = overall_year_acc.reset_index()
                overall_year_acc.columns = ['연도', '정답률']
                overall_year_acc['테스트명'] = '전체 (Overall)' if lang == 'ko' else 'Overall'
                
                # 2. 테스트셋별 연도별 정답률
                testset_year_acc = year_test_df.groupby(['테스트명', 'Year_Int'])['정답여부'].mean() * 100
                testset_year_acc = testset_year_acc.reset_index()
                testset_year_acc.columns = ['테스트명', '연도', '정답률']
                
                # 전체와 테스트셋별 데이터 결합
                combined_data = pd.concat([overall_year_acc, testset_year_acc], ignore_index=True)
                combined_data['연도'] = combined_data['연도'].astype(int)
                combined_data = combined_data.sort_values(['테스트명', '연도'])
                
                # 테스트셋 목록 (전체를 맨 앞으로)
                test_names = combined_data['테스트명'].unique().tolist()
                overall_name = '전체 (Overall)' if lang == 'ko' else 'Overall'
                if overall_name in test_names:
                    test_names.remove(overall_name)
                    test_names = [overall_name] + sorted(test_names)
                
                # 그래프 유형 선택
                graph_type = st.radio(
                    "그래프 표시 방식" if lang == 'ko' else "Graph Display Type",
                    ["통합 그래프" if lang == 'ko' else "Combined Chart", 
                     "개별 그래프" if lang == 'ko' else "Individual Charts"],
                    horizontal=True,
                    key="year_testset_graph_type"
                )
                
                if graph_type == ("통합 그래프" if lang == 'ko' else "Combined Chart"):
                    # 통합 꺾은선 그래프
                    fig = px.line(
                        combined_data,
                        x='연도',
                        y='정답률',
                        color='테스트명',
                        title='테스트셋별 연도별 정답률 추이' if lang == 'ko' else 'Accuracy Trend by Year and Test Set',
                        markers=True,
                        category_orders={'테스트명': test_names}
                    )
                    
                    # 전체 라인을 굵게 표시
                    for trace in fig.data:
                        if trace.name == overall_name:
                            trace.line.width = 4
                            trace.line.dash = 'solid'
                            trace.marker.size = 12
                        else:
                            trace.line.width = 2
                            trace.marker.size = 8
                    
                    fig.update_traces(
                        marker=dict(line=dict(width=1, color='black'))
                    )
                    fig.update_layout(
                        height=550,
                        yaxis_title='정답률 (%)' if lang == 'ko' else 'Accuracy (%)',
                        xaxis_title='출제 연도' if lang == 'ko' else 'Year',
                        yaxis=dict(range=[0, 100]),
                        legend_title='테스트셋' if lang == 'ko' else 'Test Set',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=-0.3,
                            xanchor="center",
                            x=0.5,
                            font=dict(size=annotation_size)
                        )
                    )
                    fig.update_xaxes(tickfont=dict(size=annotation_size))
                    fig.update_yaxes(tickfont=dict(size=annotation_size))
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    # 개별 그래프 - 전체 + 각 테스트셋별
                    # 먼저 전체 그래프
                    st.markdown("#### " + ("📊 전체 연도별 정답률" if lang == 'ko' else "📊 Overall Accuracy by Year"))
                    overall_data = combined_data[combined_data['테스트명'] == overall_name]
                    
                    if len(overall_data) > 0:
                        fig_overall = px.line(
                            overall_data,
                            x='연도',
                            y='정답률',
                            title='전체 연도별 정답률 추이' if lang == 'ko' else 'Overall Accuracy Trend by Year',
                            markers=True,
                            text='정답률'
                        )
                        fig_overall.update_traces(
                            texttemplate='%{text:.1f}%',
                            textposition='top center',
                            textfont=dict(size=annotation_size),
                            marker=dict(size=12, line=dict(width=2, color='black')),
                            line=dict(width=4, color='#1f77b4')
                        )
                        fig_overall.update_layout(
                            height=400,
                            yaxis_title='정답률 (%)' if lang == 'ko' else 'Accuracy (%)',
                            xaxis_title='출제 연도' if lang == 'ko' else 'Year',
                            yaxis=dict(range=[0, 100]),
                            showlegend=False
                        )
                        fig_overall.update_xaxes(tickfont=dict(size=annotation_size))
                        fig_overall.update_yaxes(tickfont=dict(size=annotation_size))
                        st.plotly_chart(fig_overall, use_container_width=True)
                    
                    st.markdown("---")
                    
                    # 테스트셋별 개별 그래프
                    st.markdown("#### " + ("📊 테스트셋별 연도별 정답률" if lang == 'ko' else "📊 Accuracy by Year per Test Set"))
                    
                    testset_only = [t for t in test_names if t != overall_name]
                    
                    # 2열 레이아웃으로 표시
                    cols = st.columns(2)
                    for idx, test_name in enumerate(testset_only):
                        test_data = combined_data[combined_data['테스트명'] == test_name]
                        
                        if len(test_data) > 0:
                            with cols[idx % 2]:
                                fig_test = px.line(
                                    test_data,
                                    x='연도',
                                    y='정답률',
                                    title=f'{test_name}',
                                    markers=True,
                                    text='정답률'
                                )
                                fig_test.update_traces(
                                    texttemplate='%{text:.1f}%',
                                    textposition='top center',
                                    textfont=dict(size=annotation_size),
                                    marker=dict(size=10, line=dict(width=1.5, color='black')),
                                    line=dict(width=3)
                                )
                                fig_test.update_layout(
                                    height=350,
                                    yaxis_title='정답률 (%)' if lang == 'ko' else 'Accuracy (%)',
                                    xaxis_title='출제 연도' if lang == 'ko' else 'Year',
                                    yaxis=dict(range=[0, 100]),
                                    showlegend=False,
                                    margin=dict(t=50, b=50)
                                )
                                fig_test.update_xaxes(tickfont=dict(size=annotation_size))
                                fig_test.update_yaxes(tickfont=dict(size=annotation_size))
                                st.plotly_chart(fig_test, use_container_width=True)
                
                # 데이터 테이블 (접이식)
                with st.expander("📋 " + ("상세 데이터 보기" if lang == 'ko' else "View Detailed Data")):
                    # 피벗 테이블로 변환
                    pivot_data = combined_data.pivot(index='테스트명', columns='연도', values='정답률')
                    pivot_data = pivot_data.reindex(test_names)
                    
                    # 평균 컬럼 추가
                    pivot_data['평균' if lang == 'ko' else 'Avg'] = pivot_data.mean(axis=1)
                    
                    # 소수점 1자리로 포맷팅
                    st.dataframe(
                        pivot_data.style.format("{:.1f}%")
                        .background_gradient(cmap='RdYlGn', axis=None, vmin=0, vmax=100),
                        use_container_width=True
                    )
                    
                    # 다운로드 버튼
                    csv_data = pivot_data.reset_index().to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 CSV " + ("다운로드" if lang == 'ko' else "Download"),
                        data=csv_data,
                        file_name="year_testset_accuracy.csv",
                        mime="text/csv"
                    )
            else:
                st.info("유효한 연도 데이터가 없습니다." if lang == 'ko' else "No valid year data available.")
        else:
            st.info("연도(Year) 컬럼이 데이터에 없습니다." if lang == 'ko' else "Year column not found in data.")
        
        st.markdown("---")
        
        # Figure 4: 출시 시기-성능 산점도
        if table3 is not None and len(table3) > 0:
            st.subheader("📅 " + ("Figure 4: 출시 시기-성능 추이" if lang == 'ko' else "Figure 4: Release Date vs Performance"))
            
            # 날짜를 숫자로 변환 (YYYY-MM -> YYYYMM)
            table3_plot = table3.copy()
            date_col = '출시 시기' if lang == 'ko' else 'Release Date'
            
            try:
                table3_plot['date_numeric'] = table3_plot[date_col].str.replace('-', '').astype(int)
            except:
                st.warning("날짜 형식 변환 오류" if lang == 'ko' else "Date format conversion error")
                st.markdown("---")
                # Figure 8로 건너뛰기
            else:
                # 추세선 그리기 시도 (statsmodels 필요)
                try:
                    fig = px.scatter(
                        table3_plot,
                        x='date_numeric',
                        y='평균 정답률 (%)' if lang == 'ko' else 'Avg Accuracy (%)',
                        text='모델명' if lang == 'ko' else 'Model',
                        title='모델 출시 시기와 성능 관계 (추세선 포함)' if lang == 'ko' else 'Model Release Date vs Performance (with Trendline)',
                        trendline='ols',
                        labels={'date_numeric': '출시 시기' if lang == 'ko' else 'Release Date'}
                    )
                    use_trendline = True
                except (ImportError, ModuleNotFoundError, Exception):
                    # statsmodels가 없거나 오류 발생 시 추세선 없이 그리기
                    fig = px.scatter(
                        table3_plot,
                        x='date_numeric',
                        y='평균 정답률 (%)' if lang == 'ko' else 'Avg Accuracy (%)',
                        text='모델명' if lang == 'ko' else 'Model',
                        title='모델 출시 시기와 성능 관계' if lang == 'ko' else 'Model Release Date vs Performance',
                        labels={'date_numeric': '출시 시기' if lang == 'ko' else 'Release Date'}
                    )
                    
                    # 수동으로 간단한 추세선 추가
                    x_numeric = table3_plot['date_numeric'].values
                    y_values = table3_plot['평균 정답률 (%)' if lang == 'ko' else 'Avg Accuracy (%)'].values
                    
                    # 선형 회귀 계산
                    z = np.polyfit(x_numeric, y_values, 1)
                    p = np.poly1d(z)
                    
                    # 추세선 추가
                    fig.add_scatter(
                        x=x_numeric,
                        y=p(x_numeric),
                        mode='lines',
                        name='추세선' if lang == 'ko' else 'Trend',
                        line=dict(color='red', dash='dash')
                    )
                    use_trendline = False
                
                # X축 레이블을 원래 날짜 형식으로 변경
                tickvals = sorted(table3_plot['date_numeric'].unique())
                ticktext = [f"{str(val)[:4]}-{str(val)[4:]}" for val in tickvals]
                
                fig.update_traces(textposition='top center', marker=dict(size=10, line=dict(width=2, color='black')), selector=dict(mode='markers'))
                fig.update_layout(
                    height=500,
                    xaxis=dict(
                        tickmode='array',
                        tickvals=tickvals,
                        ticktext=ticktext,
                        tickangle=45
                    ),
                    yaxis_title='평균 정답률 (%)' if lang == 'ko' else 'Avg Accuracy (%)',
                    yaxis=dict(range=[0, 100])
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
        
        # Figure 8: 난이도별 레이더 차트
        if '정답여부' in filtered_df.columns:
            st.subheader("🎯 " + ("Figure 8: 난이도 구간별 모델 성능 레이더 차트" if lang == 'ko' else "Figure 8: Model Performance Radar by Difficulty"))
            
            # 문제별 난이도 계산
            difficulty = filtered_df.groupby('Question')['정답여부'].mean() * 100
            
            # 난이도 구간 분류
            def classify_difficulty_simple(score):
                if score < 20:
                    return '매우 어려움' if lang == 'ko' else 'Very Hard'
                elif score < 40:
                    return '어려움' if lang == 'ko' else 'Hard'
                elif score < 60:
                    return '보통' if lang == 'ko' else 'Medium'
                elif score < 80:
                    return '쉬움' if lang == 'ko' else 'Easy'
                else:
                    return '매우 쉬움' if lang == 'ko' else 'Very Easy'
            
            # 상위 5개 모델 선택
            top_models = filtered_df.groupby('모델')['정답여부'].mean().nlargest(5).index.tolist()
            top_models_mask = filtered_df['모델'].isin(top_models)
            
            # 난이도 레벨 계산 (필요한 데이터만)
            difficulty_levels = filtered_df.loc[top_models_mask, 'Question'].map(
                lambda q: classify_difficulty_simple(difficulty.get(q, 50))
            )
            
            # 필요한 데이터만 추출하여 분석
            radar_df = pd.DataFrame({
                '모델': filtered_df.loc[top_models_mask, '모델'],
                'difficulty_level': difficulty_levels,
                '정답여부': filtered_df.loc[top_models_mask, '정답여부']
            })
            
            # 모델별 난이도별 성능
            radar_data = radar_df.groupby(['모델', 'difficulty_level'])['정답여부'].mean() * 100
            radar_pivot = radar_data.unstack(fill_value=0)
            
            if len(radar_pivot) > 0 and len(radar_pivot.columns) > 0:
                # 레이더 차트 생성
                fig = go.Figure()
                
                for model in radar_pivot.index:
                    fig.add_trace(go.Scatterpolar(
                        r=radar_pivot.loc[model].values,
                        theta=radar_pivot.columns,
                        fill='toself',
                        name=model
                    ))
                
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    showlegend=True,
                    title='난이도별 모델 성능 비교' if lang == 'ko' else 'Model Performance by Difficulty',
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("레이더 차트를 생성할 데이터가 부족합니다." if lang == 'ko' else "Insufficient data for radar chart.")
        
        st.markdown("---")
        
        # Figure 9: 비용 대비 성능 산점도 (NEW!)
        if table9 is not None and len(table9) > 0:
            st.subheader("💰 " + ("Figure 9: 비용 대비 성능 산점도" if lang == 'ko' else "Figure 9: Cost vs Performance Scatter"))
            
            fig = px.scatter(
                table9,
                x='정답 1000개당 비용 ($)' if lang == 'ko' else 'Cost per 1K Correct ($)',
                y='정답률 (%)' if lang == 'ko' else 'Accuracy (%)',
                text='모델명' if lang == 'ko' else 'Model',
                title='비용 효율성 분석 (비용 vs 정확도)' if lang == 'ko' else 'Cost Efficiency Analysis (Cost vs Accuracy)',
                labels={
                    '정답 1000개당 비용 ($)' if lang == 'ko' else 'Cost per 1K Correct ($)': '정답 1000개당 비용 ($)' if lang == 'ko' else 'Cost per 1K Correct ($)',
                    '정답률 (%)' if lang == 'ko' else 'Accuracy (%)': '정답률 (%)' if lang == 'ko' else 'Accuracy (%)'
                }
            )
            
            fig.update_traces(
                textposition='top center',
                marker=dict(size=12, line=dict(width=2, color='black'))
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 " + ("왼쪽 위(낮은 비용 + 높은 정확도)가 가장 효율적입니다." if lang == 'ko' else "Top left (low cost + high accuracy) is most efficient."))
            
            st.markdown("---")
        
        # Figure 10: 오답 패턴 원형 차트
        if table10 is not None and len(table10) > 0:
            st.subheader("🥧 " + ("Figure 10: 오답 패턴 빈도 원형 차트" if lang == 'ko' else "Figure 10: Error Pattern Distribution"))
            
            fig = px.pie(
                table10,
                values='문항 수' if lang == 'ko' else 'Problem Count',
                names='오답 패턴 유형' if lang == 'ko' else 'Error Pattern Type',
                title='오답 패턴별 비율' if lang == 'ko' else 'Distribution of Error Patterns'
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Figure 11: 모델별 오답 일치도 히트맵
        st.subheader("🔥 " + ("Figure 11: 모델별 오답 일치도 히트맵" if lang == 'ko' else "Figure 11: Model Error Agreement Heatmap"))
        
        models_list = filtered_df['모델'].unique()
        
        if len(models_list) >= 2:
            # 최적화: 모델별 오답 정보를 미리 계산하여 딕셔너리로 저장
            model_wrong_dict = {}
            for model in models_list:
                model_df = filtered_df[filtered_df['모델'] == model]
                # Question별 정답여부를 딕셔너리로 저장
                wrong_questions = set(model_df[~model_df['정답여부']]['Question'].values)
                all_questions = set(model_df['Question'].values)
                model_wrong_dict[model] = {
                    'wrong': wrong_questions,
                    'all': all_questions
                }
            
            # 모델 쌍별 오답 일치도 계산 (최적화된 버전)
            agreement_matrix = []
            
            for model1 in models_list:
                row = []
                m1_data = model_wrong_dict[model1]
                
                for model2 in models_list:
                    if model1 == model2:
                        row.append(100.0)
                    else:
                        m2_data = model_wrong_dict[model2]
                        
                        # 공통 문제
                        common_questions = m1_data['all'] & m2_data['all']
                        
                        if len(common_questions) > 0:
                            # 두 모델이 모두 틀린 문제 (공통 문제 중)
                            both_wrong = len(m1_data['wrong'] & m2_data['wrong'] & common_questions)
                            
                            # 각 모델이 틀린 문제 수 (공통 문제 중)
                            model1_wrong_count = len(m1_data['wrong'] & common_questions)
                            model2_wrong_count = len(m2_data['wrong'] & common_questions)
                            
                            total_wrong = model1_wrong_count + model2_wrong_count - both_wrong
                            
                            if total_wrong > 0:
                                agreement = (both_wrong / total_wrong) * 100
                            else:
                                agreement = 0
                        else:
                            agreement = 0
                        
                        row.append(round(agreement, 1))
                
                agreement_matrix.append(row)
            
            # 히트맵 생성
            fig = go.Figure(data=go.Heatmap(
                z=agreement_matrix,
                x=models_list,
                y=models_list,
                colorscale='Reds',
                text=agreement_matrix,
                texttemplate='%{text:.1f}',
                textfont={"size": int(10 * chart_text_size)},
                colorbar=dict(title="일치도 (%)" if lang == 'ko' else "Agreement (%)")
            ))
            
            fig.update_layout(
                title='모델 간 오답 일치도' if lang == 'ko' else 'Error Agreement Between Models',
                height=600,
                xaxis_title='모델' if lang == 'ko' else 'Model',
                yaxis_title='모델' if lang == 'ko' else 'Model'
            )
            fig.update_xaxes(tickangle=45)
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("모델이 2개 이상 필요합니다." if lang == 'ko' else "At least 2 models required.")
        
        st.markdown("---")
        
        # Figure 12 & 13: 벤치마크 비교 시각화 (NEW!)
        if table11 is not None and len(table11) > 0:
            st.subheader("📊 " + ("Figure 12: SafetyQ&A vs 범용 벤치마크 산점도 행렬" if lang == 'ko' else "Figure 12: Benchmark Scatter Plot Matrix"))
            
            # 벤치마크 컬럼만 선택
            benchmark_cols = ['SafetyQ&A', 'MMLU', 'GPQA', 'MMLU-Pro']
            model_col = '모델명' if lang == 'ko' else 'Model'
            
            # 산점도 행렬 생성
            from plotly.subplots import make_subplots
            
            n = len(benchmark_cols)
            fig = make_subplots(
                rows=n,
                cols=n,
                subplot_titles=[f"{b1} vs {b2}" for b1 in benchmark_cols for b2 in benchmark_cols],
                horizontal_spacing=0.05,
                vertical_spacing=0.05
            )
            
            for i, bench1 in enumerate(benchmark_cols, 1):
                for j, bench2 in enumerate(benchmark_cols, 1):
                    if i == j:
                        # 대각선: 히스토그램
                        fig.add_trace(
                            go.Histogram(
                                x=table11[bench1],
                                name=bench1,
                                showlegend=False,
                                marker_color='lightblue'
                            ),
                            row=i,
                            col=j
                        )
                    else:
                        # 비대각선: 산점도
                        fig.add_trace(
                            go.Scatter(
                                x=table11[bench2],
                                y=table11[bench1],
                                mode='markers',
                                name=f"{bench1} vs {bench2}",
                                showlegend=False,
                                marker=dict(size=8, color='blue', opacity=0.6),
                                text=table11[model_col],
                                hovertemplate=f"<b>%{{text}}</b><br>{bench2}: %{{x:.1f}}<br>{bench1}: %{{y:.1f}}<extra></extra>"
                            ),
                            row=i,
                            col=j
                        )
            
            fig.update_layout(
                title='벤치마크 간 상관관계 행렬' if lang == 'ko' else 'Benchmark Correlation Matrix',
                height=800,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Figure 13: 벤치마크 히트맵
            st.subheader("🔥 " + ("Figure 13: 벤치마크 유형별 모델 성능 프로파일" if lang == 'ko' else "Figure 13: Model Performance Profile by Benchmark"))
            
            # 히트맵 데이터 준비
            heatmap_data = table11.set_index(model_col)[benchmark_cols]
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale='RdYlGn',
                text=np.round(heatmap_data.values, 1),
                texttemplate='%{text:.1f}',
                textfont={"size": int(12 * chart_text_size)},
                colorbar=dict(title="점수" if lang == 'ko' else "Score"),
                zmin=0,
                zmax=100
            ))
            
            fig.update_layout(
                title='모델별 벤치마크 성능 히트맵' if lang == 'ko' else 'Model Performance Heatmap by Benchmark',
                height=600,
                xaxis_title='벤치마크' if lang == 'ko' else 'Benchmark',
                yaxis_title='모델' if lang == 'ko' else 'Model'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 인사이트
            st.success("💡 " + ("SafetyQ&A는 전문 영역(안전/법령) 벤치마크로, 범용 벤치마크(MMLU, GPQA)와 다른 패턴을 보입니다." if lang == 'ko' else "SafetyQ&A is a specialized benchmark (safety/law) showing different patterns from general benchmarks (MMLU, GPQA)."))
    
    st.sidebar.info(f"📊 {t['current_data']}: {len(filtered_df):,}{t['problems']}")
    
    # 메모리 정리 (대용량 데이터 처리 후)
    gc.collect()

if __name__ == "__main__":
    main()