---
title: "[Python] 국가별 의료비 증가와 기대수명 증가의 상관 관계 분석 1"
date: 2025-05-04
categories: [데이터분석 연습]
tags: [데이터분석, 데이터분석 연습, pandas, matplotlib, seaborn, sklearn, kmeans, python, 의료]
author: "*o^"
seo:
  type: BlogPosting
  name: "Data Lab"
---

## 국가별 의료비 증가와 기대수명 증가의 상관 관계 분석

<span class="text-blue"><strong>의료비 증가와 기대수명 증가 사이에 상관 관계가 있을까?</strong></span><br>
한국과 미국을 중심으로 전 세계 및 OECD 국가 데이터를 분석해보았습니다.<br>
의료비 지출과 기대수명 자료는 Our World In Data(링크 하단에 첨부)를 이용했습니다.

---

### 데이터 불러오기

```python
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

df = pd.read_csv("life-expectancy-vs-healthcare-expenditure.csv")
```
라이브러리와 데이터를 불러혼다.
url로 가져오는 방법도 있지만 OWID 자료는 직접 인터넷 주소창에 복붙해서 파일을 다운로드해서 가져와야 한다.

---

### 결측치 제거

```python
df_clean = df.dropna(subset=['Entity', 'life_expectancy__sex_all__age_0__variant_estimates', 'sh_xpd_chex_pp_cd'])
```
내가 보고 싶은 column에서 값이 없는 row는 "dropna"를 이용하여 제거했했다.

---

### 국가별 가장 오래된 연도와 최신 연도 데이터 추출

```python
# 국가별 오래된 연도의 정보 추출
oldest = df_clean.loc[df_clean.groupby('Entity')['Year'].idxmin()]
oldest = oldest[['Entity', 'Year', 'life_expectancy__sex_all__age_0__variant_estimates', 'sh_xpd_chex_pp_cd']]
oldest = oldest.rename(columns={
    'Year': 'Oldest_Year',
    'life_expectancy__sex_all__age_0__variant_estimates': 'LifeExp_Old',
    'sh_xpd_chex_pp_cd': 'HealthExp_Old'
})

# 국가별 최신 연도의 정보 추출
newest = df_clean.loc[df_clean.groupby('Entity')['Year'].idxmax()]
newest = newest[['Entity', 'Year', 'life_expectancy__sex_all__age_0__variant_estimates', 'sh_xpd_chex_pp_cd']]
newest = newest.rename(columns={
    'Year': 'Newest_Year',
    'life_expectancy__sex_all__age_0__variant_estimates': 'LifeExp_New',
    'sh_xpd_chex_pp_cd': 'HealthExp_New'
})

df_compare = pd.merge(oldest, newest, on='Entity')
```
얼마나 증가했는 지를 보고 싶어서 가장 오래된 연도와 최신 연도의 값들만 추출하였고, 직관적인 column 명으로 변경하였다.

---

### 증가율과 증가량 계산

```python
df_compare['LifeExp_Growth_pct'] = ((df_compare['LifeExp_New'] - df_compare['LifeExp_Old']) / df_compare['LifeExp_Old']) * 100
df_compare['HealthExp_Growth_pct'] = ((df_compare['HealthExp_New'] - df_compare['HealthExp_Old']) / df_compare['HealthExp_Old']) * 100

df_compare['LifeExp_Growth'] = df_compare['LifeExp_New'] - df_compare['LifeExp_Old']
df_compare['HealthExp_Growth'] = df_compare['HealthExp_New'] - df_compare['HealthExp_Old']
```
증가한 비율을 볼 수도 있고, 단순한 증가량을 볼 수도 있어서 두 변수 모두 구해두었다.

---

### 증가율 기준 상관관계 분석 및 시각화

```python
corr = df_compare['HealthExp_Growth_pct'].corr(df_compare['LifeExp_Growth_pct'])
print(f"상관계수: {corr}")
```
> 상관계수: **0.03** (거의 무관)

```python
plt.figure(figsize=(10, 6))

sns.regplot(
    x=df_compare['HealthExp_Growth_pct'],
    y=df_compare['LifeExp_Growth_pct']
)
plt.xlabel('의료비 증가율 (%)')
plt.ylabel('기대수명 증가율 (%)')
plt.title('의료비 증가율 vs 기대수명 증가율 (국가별)')

plt.grid(True)

# 국가명을 점 옆에 표시
for i in range(df_compare.shape[0]):
    plt.text(
        df_compare['HealthExp_Growth_pct'].iloc[i], 
        df_compare['LifeExp_Growth_pct'].iloc[i], 
        df_compare['Entity'].iloc[i], 
        fontsize=8
    )

# 한국과 미국만 강조
korea = df_compare[df_compare['Entity'] == 'South Korea']
usa = df_compare[df_compare['Entity'] == 'United States']

plt.scatter(
    korea['HealthExp_Growth_pct'],
    korea['LifeExp_Growth_pct'],
    color='red', 
    s=100, 
    label='South Korea' 
)
plt.scatter(
    usa['HealthExp_Growth_pct'],
    usa['LifeExp_Growth_pct'],
    color='blue', 
    s=100, 
    label='USA'  
)

plt.legend() # 범례 표시
plt.show()
```
![증가율 산점도](assets/img/life_medexp_rate_scatter.png)

- <strong>산점도</strong> 점들이 추세선 주변에 촘촘하게 모여 있지 않고 넓게 퍼져 있음<br>
- <strong>추세선</strong> 기울기가 거의 0, 의료비 증가율이 높아져도 기대수명 증가율이 특별히 높아지지 않음<br>
- <strong>신뢰구간(파란 그림자 부분)</strong> 폭이 넓음 → 데이터 변동성이 크고, 예측이 불확실함<br><br>
<span class="highlight-yellow"><strong>국가별로 의료비 증가율과 기대수명 증가율이 상관성이 있는가?</strong></span> 거의 무관하다.

---

### 증가량 기준 상관관계 분석 및 시각화

```python
corr = df_compare['HealthExp_Growth'].corr(df_compare['LifeExp_Growth'])
print(f"상관계수 (증가량 기준): {corr}")
```
> 상관계수: **-0.14** (거의 무관 또는 약한 음의 상관관계)

```python
plt.figure(figsize=(12, 8))
sns.regplot(
    x=df_compare['HealthExp_Growth'],
    y=df_compare['LifeExp_Growth']
)
plt.xlabel('의료비 증가량 (New - Old)')
plt.ylabel('기대수명 증가량 (New - Old)')
plt.title('의료비 증가량 vs 기대수명 증가량 (국가별)')
plt.grid(True)

korea = df_compare[df_compare['Entity'] == 'South Korea']
usa = df_compare[df_compare['Entity'] == 'United States']

plt.scatter(
    korea['HealthExp_Growth'],
    korea['LifeExp_Growth'],
    color='red',
    s=100,
    label='South Korea'
)
plt.scatter(
    usa['HealthExp_Growth_pct'],
    usa['LifeExp_Growth_pct'],
    color='blue', 
    s=100, 
    label='USA' 
)

plt.legend()
plt.show()
```
![증가량 산점도](assets/img/life_medexp_volume_scatter.png)

- <strong>산점도</strong> 왼쪽에 몰려있는 의료비가 거의 늘지 않은 국가들은 기대수명은 그래도 꽤 늘어난 경우가 많고, 오른쪽 끝의 의료비가 많이 증가한 국가들은 기대수명 증가량이 생각보다 크지 않음<br>
- <strong>추세선</strong> 음의 상관관계, 의료비 증가량이 클수록 오히려 기대수명 증가량은 줄어드는 경향성이 약하게 보임<br>
- <strong>신뢰구간</strong> 폭이 넓음 → 데이터 변동성이 크고, 이 관계가 아주 강하진 않다는 뜻<br><br>
<span class="highlight-yellow"><strong>국가별로 의료비 증가량과 기대수명 증가량이 상관성이 있는가?</strong></span> 의료비 증가량과 기대수명 증가량은 약한 음의 상관관계를 보인다. 의료비를 더 많이 쓴 나라일수록 기대수명 증가폭은 작았을 가능성이 있다.
(특히 이미 기대수명이 높은 선진국은 더 늘리기 어려움, "한계효용 체감" 현상과 비슷한 패턴)

---

### KMeans 군집화

```python
from sklearn.cluster import KMeans

X = df_compare[['HealthExp_Growth', 'LifeExp_Growth']]

kmeans = KMeans(n_clusters=3, random_state=0) 
df_compare['Cluster'] = kmeans.fit_predict(X)

print(df_compare[['Entity', 'Cluster']])
```
증가 비율(%)은 오래된 연도 값의 영향을 많이 받아 국가별 비교에 한계가 있었다. 따라서 순수한 변화 정도를 나타내는 **증가량**을 기준으로 군집화를 수행하였다.

```python
plt.figure(figsize=(12, 8))

colors = ['red', 'green', 'blue']  

for cluster in range(3):
    cluster_data = df_compare[df_compare['Cluster'] == cluster]
    plt.scatter(
        cluster_data['HealthExp_Growth'],
        cluster_data['LifeExp_Growth'],
        label=f'Cluster {cluster}',
        color=colors[cluster],
        s=80
    )

plt.xlabel('의료비 증가량 (New - Old)')
plt.ylabel('기대수명 증가량 (New - Old)')
plt.title('국가별 의료비 & 기대수명 증가량 군집화 결과')
plt.legend()
plt.grid(True)
plt.show()
```
![증가량 기준 군집화](assets/img/life_medexp_clustering.png)

이번 군집화 결과에서는 의료비 증가량과 기대수명 증가량의 관계에 따라 국가들이 세 그룹으로 나뉘었다.

- **Cluster 0 (빨강)**  
  의료비 증가량이 낮은 국가들이며, 기대수명 증가량은 국가별로 크게 다르게 나타나 분산이 큰 집단이다. 이 그룹은 의료비를 늘리지 않고도 기대수명이 증가했거나, 다른 사회적·보건적 요인에 의해 건강 성과가 개선된 국가들을 포함한다.

- **Cluster 1 (초록)**  
  의료비 증가 높음, 기대수명 증가 중간

- **Cluster 2 (파랑)**  
  의료비 증가 중간, 기대수명 증가 중간

**한국**은 높은 의료비 증가와 함께 기대수명 증가도 상대적으로 높아 **효율적인 투자 효과**를 보여주고 있다.  
반면, **미국**은 의료비 증가 폭은 가장 크지만 기대수명 증가량은 상대적으로 낮아, **비효율적인 투자 사례**로 해석된다.

이번 결과는 국가별 보건 재정 투입과 건강 성과 사이의 관계를 군집화 방법으로 시각화함으로써, 단순 상관계수 분석 이상의 통찰을 제공해주었다.

---

### OECD 국가 필터링

전 세계 국가들은 경제력, 의료 접근성, 인구 구조 등이 너무 다양해서 의료비와 기대수명 데이터를 단순 비교하기에는 어려운 것 같아 OECD 국가 필터링을 해보았다.

```python
oecd_countries = [
    'Australia', 'Austria', 'Belgium', 'Canada', 'Chile', 'Colombia', 'Czechia', 'Denmark',
    'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland',
    'Israel', 'Italy', 'Japan', 'South Korea', 'Latvia', 'Lithuania', 'Luxembourg', 'Mexico',
    'Netherlands', 'New Zealand', 'Norway', 'Poland', 'Portugal', 'Slovakia', 'Slovenia',
    'Spain', 'Sweden', 'Switzerland', 'Turkey', 'United Kingdom', 'United States'
]

df_oecd = df_compare[df_compare['Entity'].isin(oecd_countries)]

for cluster in range(3):
    countries = df_oecd[df_oecd['Cluster'] == cluster]['Entity'].tolist()
    print(f"\n===== Cluster {cluster} : OECD 국가 (총 {len(countries)}개) =====")
    print(countries)
```
>===== Cluster 0 : OECD 국가 (총 3개) =====<br>
>['Colombia', 'Mexico', 'Turkey']
>
>===== Cluster 1 : OECD 국가 (총 17개) =====<br>
>['Australia', 'Austria', 'Belgium', 'Canada', 'Denmark', 'Finland', 'France', 'Germany', 'Ireland', 'Luxembourg', 'Netherlands', 'Norway', 'South Korea', 'Sweden', 'Switzerland', 'United Kingdom', 'United States']
>
>===== Cluster 2 : OECD 국가 (총 16개) =====<br>
>['Chile', 'Czechia', 'Estonia', 'Greece', 'Hungary', 'Iceland', 'Israel', 'Italy', 'Japan', 'Latvia', 'Lithuania', 'New Zealand', 'Poland', 'Portugal', 'Slovenia', 'Spain']

각 군집에 들어가는 국가명 리스트를 출력하여 해당 국가들의 특징을 떠올려보았다.

Cluster 0은 의료비 증가가 적고 기대수명 증가도 제한적인 국가들, Cluster 1은 높은 의료비 증가와 평균적인 기대수명 증가를 보인 선진국들, 그리고 Cluster 2는 중간 정도의 의료비 증가로 효율적인 기대수명 증가를 달성한 국가들로 나눌 수 있었다.

---

### OECD 국가 군집화 시각화

```python
plt.figure(figsize=(12, 8))

colors = ['red', 'green', 'blue']

for cluster in range(3):
    cluster_data = df_oecd[df_oecd['Cluster'] == cluster]
    plt.scatter(
        cluster_data['HealthExp_Growth'],
        cluster_data['LifeExp_Growth'],
        label=f'Cluster {cluster}',
        color=colors[cluster],
        s=100
    )

plt.xlabel('의료비 증가량 (New - Old)')
plt.ylabel('기대수명 증가량 (New - Old)')
plt.title('OECD 국가 의료비 & 기대수명 증가량 군집화 결과')
plt.legend()
plt.grid(True)


for i in range(df_oecd.shape[0]):
    plt.text(
        df_oecd['HealthExp_Growth'].iloc[i],
        df_oecd['LifeExp_Growth'].iloc[i],
        df_oecd['Entity'].iloc[i],
        fontsize=8
    )

plt.show()
```
![증가량 기준 OCED 필터링 군집화](assets/img/life_medexp_clustering_oecd.png)

군집화한 산점도에서 한국의 위치를 보면 <span class="text-red"><strong>한국은 OECD 평균을 넘어서며, 의료비 증가량이 컸지만 기대수명 증가에서 비용대비 효율적 성과를 달성</strong></span>했다.<br>

```python
df_oecd_sorted = df_oecd.sort_values(by='HealthExp_New', ascending=False).reset_index(drop=True)

korea_rank = df_oecd_sorted[df_oecd_sorted['Entity'] == 'South Korea'].index[0] + 1

print(f"한국의 의료비 순위: {korea_rank}위 (총 {len(df_oecd_sorted)}개국 중)")
```
> 한국의 의료비 순위: 18위 (총 36개국 중)

최신 연도의 의료비 지출 기준으로 한국은 OECD 36개국 중 18위에 해당한다. 의료비 증가량만 놓고 보면 한국은 상위 그룹에 속한다. 기대수명 증가량은 OECD 36개국 중 2위이다. 이러한 결과는 한국의 의료비 지출 증가가 단순한 비용 증가에 그치지 않고, 실제로 국민 건강 향상에도 긍정적인 영향을 주었음을 시사한다.

---

## 📊 인사이트

- **전 세계 데이터** 의료비 증가와 기대수명 증가의 관계는 거의 없음
- **OECD 국가**
  - **Cluster 0** 의료비 증가 낮음, 기대수명 증가는 국가별 편차가 큼
  - **Cluster 1** 의료비 증가 높음, 기대수명 증가 중간 (한국 포함)
  - **Cluster 2** 의료비 증가 중간, 기대수명 증가 중간
- **대한민국** 비교적 높은 의료비 증가량 대비 기대수명 증가 성과 준수
- **미국** 의료비 증가량 최고이나 성과 낮음

---

_분석에 사용한 데이터: Our World in Data [OWID](https://ourworldindata.org/)_  
_전체 코드 및 데이터 [GitHub 링크](https://github.com/yyuseon/data_analysis_practice)_
<br><br>
