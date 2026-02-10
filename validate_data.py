"""
데이터 검증 및 간단한 통계 출력 스크립트
"""

import pandas as pd
import glob
import os
from pathlib import Path

def validate_data(data_dir="/mnt/project"):
    """데이터 검증 및 기본 통계 출력"""
    
    print("="*80)
    print("🔍 LLM 벤치마크 데이터 검증")
    print("="*80)
    print()
    
    # Testset 파일 확인
    print("📚 Testset 파일:")
    print("-"*80)
    testset_files = glob.glob(os.path.join(data_dir, "testset_*.csv"))
    
    if not testset_files:
        print("⚠️  Testset 파일을 찾을 수 없습니다!")
        return
    
    testsets = {}
    for file in testset_files:
        test_name = os.path.basename(file).replace("testset_", "").replace(".csv", "")
        try:
            df = pd.read_csv(file, encoding='utf-8')
        except:
            df = pd.read_csv(file, encoding='cp949')
        
        testsets[test_name] = df
        print(f"  ✓ {test_name}: {len(df)} 문제")
        
        # 법령 문제 수 확인
        if 'law' in df.columns:
            law_count = (df['law'] == 'O').sum()
            print(f"    - 법령 문제: {law_count} ({law_count/len(df)*100:.1f}%)")
        
        # 과목 확인
        if 'Subject' in df.columns:
            subjects = df['Subject'].nunique()
            print(f"    - 과목 수: {subjects}")
    
    print()
    
    # 결과 파일 확인
    print("📊 결과 파일:")
    print("-"*80)
    
    result_files = glob.glob(os.path.join(data_dir, "*_detailed_*.csv")) + \
                   glob.glob(os.path.join(data_dir, "*_summary_*.csv"))
    
    if not result_files:
        print("⚠️  결과 파일을 찾을 수 없습니다!")
        return
    
    results_summary = {}
    
    for file in result_files:
        filename = os.path.basename(file)
        
        # 파일명 파싱
        if "Claude" in filename:
            if "3-5-Sonnet" in filename:
                model = "Claude-3.5-Sonnet"
            elif "3-5-Haiku" in filename:
                model = "Claude-3.5-Haiku"
            else:
                continue
        elif "GPT" in filename:
            if "4o-Mini" in filename:
                model = "GPT-4o-Mini"
            elif "4o" in filename:
                model = "GPT-4o"
            else:
                continue
        else:
            continue
        
        try:
            df = pd.read_csv(file, encoding='utf-8')
        except:
            try:
                df = pd.read_csv(file, encoding='cp949')
            except:
                continue
        
        if model not in results_summary:
            results_summary[model] = {
                'files': 0,
                'total_questions': 0,
                'correct': 0,
                'accuracy': []
            }
        
        results_summary[model]['files'] += 1
        results_summary[model]['total_questions'] += len(df)
        
        if '정답여부' in df.columns:
            results_summary[model]['correct'] += df['정답여부'].sum()
            results_summary[model]['accuracy'].append(df['정답여부'].mean() * 100)
    
    # 모델별 통계 출력
    for model, stats in sorted(results_summary.items()):
        print(f"\n  {model}:")
        print(f"    - 결과 파일 수: {stats['files']}")
        print(f"    - 총 문제 수: {stats['total_questions']}")
        print(f"    - 정답 수: {stats['correct']}")
        
        if stats['accuracy']:
            avg_acc = sum(stats['accuracy']) / len(stats['accuracy'])
            print(f"    - 평균 정확도: {avg_acc:.2f}%")
            print(f"    - 최고 정확도: {max(stats['accuracy']):.2f}%")
            print(f"    - 최저 정확도: {min(stats['accuracy']):.2f}%")
    
    print()
    print("="*80)
    print("✅ 데이터 검증 완료!")
    print("="*80)
    print()
    print("💡 다음 명령으로 시각화 도구를 실행하세요:")
    print("   streamlit run llm_benchmark_visualizer.py")
    print()
    print("   또는:")
    print("   ./run.sh")
    print()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "/mnt/project"
    
    if not os.path.exists(data_dir):
        print(f"❌ 경로를 찾을 수 없습니다: {data_dir}")
        sys.exit(1)
    
    validate_data(data_dir)
