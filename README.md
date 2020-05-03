# Titanic-Machine-Learning-from-Disaster
캐글 타이타닉 이유한님 유튜브 영상 참고
### Data Science Process
&nbsp;데이터 사이언스는 다음과 같은 단계로 구성되어 있다.
1. Project Scoping(Define Problem)
2. Data Collection
3. Dataset Check
4. EDA(Exploratory Data Analysis)
5. Data Preprocessing
6. Feature Engineering
7. Modeling
8. Evaluation
9. Project Delivery / Insights 
<p>하지만 이 레포지터리에서는 캐글의 타이타닉 생존자 예측을 하는 것이기 때문에 문제가 정의되어 있고 데이터도 모여있다. 따라서 1, 2번 과정을 스킵한다. 그리고 Data Preprocessing과 Feature Engineering은 동일시하는 경향이 있지만 엄밀한 의미에서 보면 최종 목적이 다르기 때문에 목적에 맞게 구분하는 것을 권장한다고 하지만 내가 차이를 이해하지 못했기 때문에 여기선 동일하다고 가정하겠다. 그렇게 되면 우리가 최종적으로 하게될 것은 EDA, Feature Engineering, Modeling, Evaluation, Project Delivery / Insight 가 되겠다.</p>
&nbsp;이제부터 캐글 타이타닉 생종자 예측을 시작해 보도록 하겠다.

### Modules
```python
import numpy as np # 행렬 연산을 위한 모듈
import pandas as pd # 데이터프레임을 위한 모듈

import matplotlib.pyplot as plt # 그래프를 그리기 위한 모듈
import seaborn as sns # 이놈도 그래프를 그리기 위한 모듈

import missingno as msno # NULL 데이터 쉽게 보여주는 모듈

%matplotlib inline # 주피터 노트북에서 바로 그래프같은 Rich output를 볼 수 있도록 하는 구문
```
```python
# 캐글의 데이터셋 변수에 저장
df_train = pd.read_csv("train.csv주소")
df_test = pd.read_csv("test.csv주소")
```

### Dataset Check
&nbsp;위에서 필요한 모듈을 다 불러오고 데이터셋도 변수에 저장했으므로 이제 데이터셋을 확인해보도록 하자<br>
&nbsp;우선 데이터셋의 feature들이 각각 어떤걸 의미하는 지 볼 필요가 있다. 데이터셋에 대한 설명은 캐글에 있다.
<img src="https://media.discordapp.net/attachments/706368531175964732/706381469404102676/unknown.png" title="Dataset" alt="Dataset"></img><br>
```python
df_train.head()
```
<img src="https://media.discordapp.net/attachments/706368531175964732/706368547483549716/unknown.png" title="df_train.head()" alt="df_train.head()"></img><br>
&nbsp;이 문제에서 feature는 Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked이고 target label은 Survived이다.
```python
df_train.describe()
```
<img src="https://media.discordapp.net/attachments/706368531175964732/706369761004617768/unknown.png" title="df_train.describe()" alt="df_train.describe()">
&nbsp;pandas dataframe 에는 describe() 메소드가 있는 데, 이를 쓰면 수치형 feature 가 가진 통계치들을 반환해준다.

#### Null Data Check
```python
for col in df_train.columns:
  msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (df_train[col].isnull().sum() / df_train[col].shape[0]))
  print(msg)
```
<img src="https://media.discordapp.net/attachments/706368531175964732/706374446226604083/unknown.png" title="Null Check" alt="Null Check"></img><br>
&nbsp;다음과 같은 코드로 각 컬럼마다의 결측치를 확인할 수 있었다. 이러한 방법 말고도 missingno 모듈을 이용해서 결측치를 그래프로 살펴볼 수도 있다.
```python
msno.matrix(df=df_train.iloc[:, :], figsize=(8, 8))
```
<img src="https://media.discordapp.net/attachments/706368531175964732/706375461680185350/pki6RdHpT8yZJr0uaZFlWTZT37yxpawtl9GsalgEAAA77TvtEAAAABLV2TQQAAEAYQgQAADgphAgAAHBSCBEAAOCkECIAAMBJIUQ.png" title="msno matrix Null Check" alt="msno matrix Null Check"></img><br>
```python
msno.bar(df=df_train.iloc[:, :], figsize=(8, 8))
```
<img src="https://media.discordapp.net/attachments/706368531175964732/706376759880319026/oWPKiEhISEhISFhPrHAvZ8iFXUmcg00t1HFjmwhISEhISEhIT5QY0aWkYzy0OBK1N1U0JCQkJCQkJdosZdujOksu2EhISEhISEuk.png" title="msno bar Null Check" alt="msno bar Null Check"></img><br>

#### Target Label Check
&nbsp;데이터셋을 확인할 때에 target label이 어떤 분포를 가지고 있는지 확인해야 한다. 만약 지금과 같은 이진분류 문제로 데이터셋이 100 중 99:1의 비율을 갖고 있다고 가정할 때, 모델이 모든 예측을 99의 비율을 갖고있는 target label로 예측을 한다고 해도 99%의 정확도를 가지게 되므로 원하는 결과를 얻을 수 없게 될 수 있다. 그래서 만약 데이터셋이 균일하지 않다면(imbalanced) resampling등과 같은 방법을 통해 학습을 시켜야한다.
```python
f, ax = plt.subplots(1, 2, figsize=(18, 8))

df_train["Survived"].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
ax[0].set_title("Pie plot - Survived")
ax[0].set_ylabel('')
sns.countplot("Survived", data=df_train, ax=ax[1])
ax[1].set_title("Count plot - Survived")
plt.show()
```
<img src="https://media.discordapp.net/attachments/706368531175964732/706379103489490984/B1Ms4PxiXnApAAAAAElFTkSuQmCC.png" title="Target Label Distribution Check" alt="Target Label Distribution Check"></img><br>
&nbsp;target label의 분포를 살펴본 결과 제법 균일(balanced)하게 나왔기 때문에 그냥 학습을 진행해도 괜찮아 보인다.

### EDA(Exploratory Data Analysis)
#### EDA의 정의
&nbsp;EDA란 수집한 데이터가 들어왔을 때, 이를 다양한 각도에서 관찰하고 이해하는 과정이다. 즉 데이터를 분석하기 전, 그래프나 통계적인 방법으로 자료를 직관적으로 바라보는 과정이다.<br>
<br>
본격적으로 데이터를 살펴보기 전 http://blog.heartcount.io/dd 이 블로그에 들어가 데이터의 유형에 대해 살펴보면 좋을 것 같다.

#### Pclass
&nbsp;들어가기 전 Pclass는 서수형 데이터(ordinal data)이다. 즉 카테고리이면서 순서가 있는 데이터 유형이다.
```python
pd.crosstab(df_train['Pclass'], df_train['Survived'], margins=True).style.background_gradient(cmap='summer_r')
```
<img src="https://media.discordapp.net/attachments/706368531175964732/706383604489650206/unknown.png" title="crosstab Pclass & Survived" alt="crosstab Pclass & Survived"></img><br>
pandas의 crosstab를 통해 클래스별 생존자 / 사망자를 확인해 볼 수 있다. 여기서 Pclass가 1이면 136 / 216으로 약 63%의 생존률을 갖는다는 것을 알 수 있다.<br>
&nbsp;이를 matplotlib으로 시각화 해 보면 다음과 같이 나온다.
```python
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar()
```
<img src="https://media.discordapp.net/attachments/706368531175964732/706390011196735498/eFOiKIAeQosAAAAASUVORK5CYII.png" title="Survived-Pclass survival rate" alt="Survived-Pclass survival rate"></img><br>
&nbsp;또 seaborn라이브러리를 통해 Pclass당 승객 수, 생존 / 사망을 시각화 해보면 다음과 같이 나온다.
```python
y_position = 1.02
f, ax = plt.subplots(1, 2, figsize=(18, 8))
df_train['Pclass'].value_counts().plot.bar(ax=ax[0])
ax[0].set_title('Number of passengers By Pclass', y=y_position)
ax[0].set_ylabel('Count')
sns.countplot('Pclass', hue='Survived', data=df_train, ax=ax[1])
ax[1].set_title('Pclass: Survived vs Dead', y=y_position)

plt.show()
```
<img src="https://cdn.discordapp.com/attachments/706368531175964732/706385489376182272/wPOEu9mTDUlnwAAAABJRU5ErkJggg.png" title="sns graph Pclass & Survived" alt="sns graph Pclass & Survived"></img><br>
&nbsp;여기서 얻을 수 있는 결론은 Pclass가 좋을수록 생존률이 높다는 것이다. 따라서 우리는 생존에 Pclass가 큰 영향을 미친다고 생각해 볼 수 있으며 이를 feature로 사용하는 것은 좋은 판단이라는 결론을 얻을 수 있다.

#### Sex
&nbsp;들어가기 전 Sex는 범주형 데이터(categorical data)이다.
&nbsp;이번에는 성별해 따라 생존률이 어떻게 달라지는 지 살펴볼 것이다. sns로 시각화 해 보면 다음과 같이 나온다.
```python
f, ax = plt.subplots(1, 2, figsize=(18, 8))
df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=True).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot('Sex', hue='Survived', data=df_train, ax=ax[1])
ax[1].set_title('Sex: Survived vs Dead')

plt.show()
```
<img src="https://media.discordapp.net/attachments/706368531175964732/706391680986775602/2unnLUmSJKm3mMyQJEmSJEmNYgFQSZIkSZLUKCYzJEmSJElSo5jMkCRJkiRJjWIyQ5IkSZIkNYrJDEmSJEmS1CgmMyRJkiRJUqOY.png" title="Sex & Survival graph" alt="Sex & Survival graph"></img><br>
&nbsp;crosstab으로 그려보면 다음과 같이 나온다.
```python
pd.crosstab(df_train['Sex'], df_train['Survived'], margins=True).style.background_gradient(cmap='summer_r')
```
<img src="https://media.discordapp.net/attachments/706368531175964732/706392280373919794/unknown.png" title="Sex & Survival crosstab" alt="Sex & Survival crosstab"></img><br>
&nbsp;여기서 얻을 수 있는 결론은 여성이 생존확률이 더 높다는 것이다. 따라서 이 또한 feature로 사용하는 것이 좋은 판단이라고 할 수 있다.

#### Sex and Pclass
&nbsp;이번에는 Sex, Pclass가 Survived에 대해 어떻게 달라지는 지 확인해 볼 것이다.<br>
&nbsp;seaborn의 factorplot을 이용하면 3개의 차원으로 이루어진 그래프를 그릴 수 있다. 
```python
sns.factorplot('Pclass', 'Survived', hue='Sex', data=df_train, size=6, aspect=1.5)
```
<img src="https://cdn.discordapp.com/attachments/706368531175964732/706392562336006144/14Npp4PdDCL8kCQ8LSFZK3wmsBm6PMd4BEEI40aUuIhm8vhioAmaQtHC0ArsyNd4LfKTqXkfCiFckHn5cD5JFP6DiMZt7IVeAL4b.png" title="Sex & Pclass & Survival factorplot" alt="Sex & Pclass & Survival factorplot"></img><br>
위의 그래프를 살펴보면 모든 클래스에서 여성이 살 확률이 남성보다 높다는 것을 알 수 있고, 남자, 여자 상관없이 클래스가 높을 수록 살 확률이 높다는 것을 알 수 있다.

### 참고자료
http://hero4earth.com/blog/learning/2018/01/29/Feature_Engineering_Basic/ <br>
https://eda-ai-lab.tistory.com/13 <br>
https://statkclee.github.io/model/model-feature-engineering.html <br>
https://www.youtube.com/watch?v=_-N-kdodS0o&list=PLC_wC_PMBL5MnqmgTLqDgu4tO8mrQakuF&index=1 <br>
