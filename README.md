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
9. Project Delivery / Insights (Submission)
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

#### Age
&nbsp;이번에는 Age feature를 살펴보도록 하겠다.
```python
print("제일 나이 많은 탑승객 : {:.1f} years".format(df_train['Age'].max()))
print("제일 어린 탑승객 : {:.1f} years".format(df_train['Age'].min()))
print("탑승객 평균 나이 : {:.1f} years".format(df_train['Age'].mean()))
```
<img src="https://media.discordapp.net/attachments/706368531175964732/706398111542804520/unknown.png" title="Age stat" alt="Age stat"></img><br>
&nbsp;생존에 따른 Age의 kdeplot을 그려보면 다음과 같이 나온다.
```python
fig, ax = plt.subplots(1, 1, figsize=(9, 5))
sns.kdeplot(df_train[df_train['Survived'] == 1]['Age'], ax=ax)
sns.kdeplot(df_train[df_train['Survived'] == 0]['Age'], ax=ax)
plt.legend(['Survived == 1', 'Survived == 0'])

plt.show()
```
<img src="https://cdn.discordapp.com/attachments/706368531175964732/706399881144696832/H9FRTV31UTl1wAAAABJRU5ErkJggg.png" title="Age & Survived kdeplot" alt="Age & Survived kdeplot"></img><br>
위 그래프를 살펴보면 생존자 중 나이가 어린 경우가 많음을 알 수 있다.
&nbsp;Pclass당 나이 분포를 살펴보면 아래와 같이 그려진다.
```python
plt.figure(figsize=(8, 6))
df_train['Age'][df_train['Pclass'] == 1].plot(kind='kde')
df_train['Age'][df_train['Pclass'] == 2].plot(kind='kde')
df_train['Age'][df_train['Pclass'] == 3].plot(kind='kde')

plt.xlabel('Age')
plt.title('Age Distribution within classes')
plt.legend(['1st Class', '2nd Class', '3rd Class'])

plt.show()
```
<img src="https://media.discordapp.net/attachments/706368531175964732/706402784232341514/8HBqkuXp0PwGoAAAAASUVORK5CYII.png" title="Age Distribution within classes" alt="Age Distribution within classes"></img><br>
위 그래프를 살펴보면 Pclass가 높을수록 나이 많은 사람의 비중이 커짐을 알 수 있다.

&nbsp;이번에는 나이대가 변하면서 생존률이 어떻게 변화하는 지 보기 위 해 나이 범위를 점점 넓혀가며, 생존률이 어떻게 되는지 살펴볼 것이다.
```python
change_age_range_survival_ratio = []

for i in range(1, 80):
  change_age_range_survival_ratio.append(df_train[df_train['Age'] < i]['Survived'].sum() / len(df_train[df_train['Age'] < i]['Survived']))

plt.figure(figsize=(7, 7))
plt.plot(change_age_range_survival_ratio)
plt.title('Survival rate change depending on range of Age', y=1.02)
plt.ylabel('Survival rate')
plt.xlabel('Range of Age(0~x)')

plt.show()
```
<img src="https://media.discordapp.net/attachments/706368531175964732/706401578537451611/W0RkYNSSLiIiIiIyxKjjqIiIiIjIEKMgXURERERkiFGQLiIiIiIyxChIFxEREREZYhSki4iIiIgMMQrSRURERESGmP8PPWvSwYkV.png" title="Survival rate change depending on range of Age" alt="Survival rate change depending on range of Age"></img><br>
위 그래프를 살펴보면 나이가 어릴 수록 생존률이 확실히 높은 것을 확인할 수 있다. 따라서 Age는 중요한 feature로 사용될 수 있다.

#### Pclass, Sex and Age
&nbsp;여태까지 본 Sex, Pclass, Age, Survived 모두에 대해 보고 싶으면 seaborn의 violinplot을 사용하는 것도 하나의 방법이다. x축을 Pclass, Sex로 y축을 Age로 두고 그래프를 그리면 다음과 같이 나온다.
```python
f, ax = plt.subplots(1, 2, figsize=(18, 8))
sns.violinplot('Pclass', 'Age', hue='Survived', data=df_train, scale='count', split=True, ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')

sns.violinplot('Sex', 'Age', hue='Survived', data=df_train, scale='count', split=True, ax=ax[1])
ax[0].set_title('Sex and Age vs Survived')

plt.show()
```
<img src="https://cdn.discordapp.com/attachments/706368531175964732/706408099694641172/w9WbOsO9iJHtwAAAABJRU5ErkJggg.png" title="violin plot" alt="violin plot"></img><br>
생존만 봤을 때. 모든 클래스에서 나이가 어릴 수록 생존을 많이 한것을 볼 수 있다. 오른쪽 그래프를 보면 명확히 여자가 많이 생존한 것을 볼 수 있다. 결과적으로 여성과 아이를 먼저 챙긴 것을 알 수 있다.

#### Embarked
&nbsp;Embarked는 탑승한 항구를 나타낸다. 이번에는 탑승한 곳에 따른 생존률을 보겠다.
```python
f, ax = plt.subplots(1, 1, figsize=(7, 7))
df_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax)
```
<img src="https://cdn.discordapp.com/attachments/706368531175964732/706416133103681577/adAAAAAElFTkSuQmCC.png" title="Embarked Survial rate" alt="Embarked Survival rate"></img><br>
위 그래프를 보면 항구마다 조금 씩 차이는 있지만 생존률은 비슷하다. 이 feature가 모델에 얼마나 큰 영향을 미칠지는 모르겠지만 그래도 사용하도록 한다. 미리 스포일러 하자면 RandomForest 기준 영향 미친건 쥐똥만하다.
&nbsp;Embarked를 다른 feature와 함께 그래프를 그리면 다음과 같이 그려진다.
```python
f, ax = plt.subplots(2, 2, figsize=(20, 15))
sns.countplot('Embarked', data=df_train, ax=ax[0, 0])
ax[0, 0].set_title('(1) No. Of Passengers Boared')

sns.countplot('Embarked', hue='Sex', data=df_train, ax=ax[0, 1])
ax[0, 1].set_title('(2) Male-Female split for embarked')

sns.countplot('Embarked', hue='Survived', data=df_train, ax=ax[1, 0])
ax[1, 0].set_title('(3) Embarked vs Survived')

sns.countplot('Embarked', hue='Pclass', data=df_train, ax=ax[1, 1])
ax[1, 1].set_title('(4) Embarked vs Pclass')

plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()
```
<img src="https://media.discordapp.net/attachments/706368531175964732/706417747994607646/BeoOtZn5wWCNAAAAAElFTkSuQmCC.png?width=900&height=677" title="Emabrked and various features" alt="Emabrked and various features"></img><br>
첫 번째 그림을 봤을 때 S에서 가장 많은 사람이 탑승했다. <br>
두 번째 그림을 봤을 때에는 C와 Q의 남녀비율이 비슷하고 S는 남자가 더 많다.<br>
세 번째 그림을 보면 생존확률이 S의 경우 많이 낮은 걸 볼 수 있다. 아마 남자의 비율이 높았기 때문일지도 모른다. <br>
네 번째 그림에서는 C가 생존확률이 높은 것은 클래스가 높은 사람이 많이 타서 그런것으로 보인다. 그리고 S는 3rd class가 많아서 생존확률이 낮게 나오는 것 같다.

#### FamilySize(SibSp(형제 자매) + Parch(부모 자녀) + 1(나))
&nbsp; SibSp와 Parch를 합치면 Family가 될 것이다. Family로 합쳐서 분석해보도록 한다.
```python
df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1
df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1
```
```python
print('Maximum size of Family : ', df_train['FamilySize'].max())
print('Minimum size of Family : ', df_train['FamilySize'].min())
```
위의 코드로 가장 큰 가족과 작은 가족을 찾으면 11, 1이 나온다.<br>
&nbsp;FamilySize와 생존의 관계를 살펴보면 다음과 같다.
```python
f, ax = plt.subplots(1, 3, figsize=(40, 10))
sns.countplot('FamilySize', data=df_train, ax=ax[0])
ax[0].set_title('(1) No. Of Passenger Boarded', y=1.02)

sns.countplot('FamilySize', hue='Survived', data=df_train, ax=ax[1])
ax[1].set_title('(2) Survived countplot depending on FamilySize', y=1.02)

df_train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax[2])
ax[2].set_title('(1) No. Of Passenger Boarded', y=1.02)

plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()
```
<img src="https://media.discordapp.net/attachments/706368531175964732/706421155384524870/rJA4tCkobUmoMOQJIkSZIkSZIkSZKkqSYitgW2BV4CzK8LwOWZefXAApOkIWUnJUmSJEmSJEmSJEmSxu5g4LimfY8AHx9ALJI09C.png?width=1442&height=417" title="(1) No. Of Passenger Boarded (2) Survived countplot depending on FamilySize (3) No. Of Passenger Boarded" alt="(1) No. Of Passenger Boarded (2) Survived countplot depending on FamilySize (3) No. Of Passenger Boarded"></img><br>
(1)그림을 살펴보면 가족 크기는 1 ~ 11까지 있고 대부분 1명, 그 다음으로는 2, 3, 4명인 걸 알 수 있다.<br>
(2), (3)그림을 살펴보면 가족이 4명인 경우가 가장 생존 확률이 높다. 가족수가 너무 많아도 너무 적어도 생존 확률이 작아진다. 3 ~ 4명 선에서 생존 확률이 높은 것을 확인할 수 있다.

#### Fare
Fare는 탑승 요금이며 연속형 feature이다.
```python
f, ax = plt.subplots(1, 1, figsize=(8, 8))
g = sns.distplot(df_train['Fare'], color='b', label='Skewness: {:.2f}'.format(df_train['Fare'].skew()), ax=ax)
g = g.legend(loc='best')
```
<img src="https://media.discordapp.net/attachments/706368531175964732/706422927704326204/Acgecs91zgiqAAAAAElFTkSuQmCC.png" title="Skewness of Fare" alt="Skewness of Fare"></img><br>
Fare를 distplot으로 그려 비대칭도를 살펴보면 왼쪽으로 매우 비대칭인 것을 알 수 있다. 만약 데이터가 이렇게 비대칭인 체 모델에 넣어준다면 outlier에 매우 민감하게 반응하는 모델이 만들어 질 수 있다. 이것을 대칭으로 만들어 주는 것은 Feature Engineering 때 하도록 하겠다.

#### Cabin
&nbsp;이 feature는 NaN이 너무 많으므로 생존에 영향을 미칠 중요한 정보를 얻어내기가 어렵다. 그러므로 모델에 포함시키지 않도록 하겠다.

#### Ticket
&nbsp; 이 feature는 Nan이 없으나 string data이므로 우리가 어떤 작업들을 해주어야 실제 모델에 사용할 수 있다. 이를 위해선 아이디어가 필요하다. 이유한님의 강의에서는 다루지않고 직접해보라고 하였으므로 넘어가도록 하겠다.

### Feature Engineering

#### Feature Engineering 정의
&nbsp;Feature Engineering은 머신러닝 알고리즘을 작동하기 위해 데이터에 대한 도메인 지식을 활용하여 feature를 만들어내는 작업이다. 다시말해 모델의 성능을 높이기 위해 모델에 입력할 데이터를 만들기 주어진 초기 데이터로부터 특징을 가공하고 생성하는 전체 과정을 의미한다.

#### Fill Null
&nbsp;결측치를 채울 때 주의할 점이 있는데, test와 train의 결측치는 오직 train의 데이터로만 추정하여 넣어야 한다.
&nbsp;train데이터의 Age칼럼에는 Null data가 177개 있다. 이를 채우는 방법으로 title당 평균으로 채우는 방법으로 할 것이다. 영어에는 Mr., Mrs.,  Miss, Master(Mstr.)의 호칭이 있다. Mr.는 성인 남성, Mrs.는 결혼을 한 여성, Miss는 결혼을 하지 않은 여성, Master는 결혼을 하지 않은 남성 중 주로 청소년 이하를 지칭한다. 각 탑승객의 이름에는 꼭 이런 title이 들어가게 된다. 따라서 이 title을 이용해 보도록 하겠다.<br>
&nbsp;이러한 타이틀을 python의 extract를 통해 initial을 추출해 보도록 하겠다.
```python
df_train['Initial'] = df_train['Name'].str.extract('([A-Za-z]+)\.')
df_test['Initial'] = df_test['Name'].str.extract('([A-Za-z]+)\.')
```
```python
pd.crosstab(df_train['Initial'], df_train['Sex']).T.style.background_gradient(cmap='summer_r')
```
<img src="https://media.discordapp.net/attachments/706368531175964732/706431522143731722/unknown.png" title="crosstab of initial" alt="crosstab of initial"></img><br>
추출한 Initial을 crosstab을 통해 성별에 따라 나눠서 보면 위와 같은 그림이 나온다.<br>
&nbsp;여기서 나온 title을 모두 Master, Miss, Mr, Mrs, Other로 치환해 준다. 그 코드를 작성하면 다음과 같이 된다.
```python
df_train['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)

df_test['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)
```
```python
df_train.groupby('Initial')['Survived'].mean().plot.bar()
```
<img src="https://cdn.discordapp.com/attachments/706368531175964732/706439936982908968/WJZ97okLAAAAAAElFTkSuQmCC.png" title="mean of each title graph" alt="mean of each title graph"></img><br>
위 표를 살펴보면 여성과 관계있는 Miss, Mr가, 아이와 관계있는 Master가 생존률이 높은 것을 볼 수 있다.
```python
df_train.groupby('Initial')['Survived'].mean()
```
<img src="https://media.discordapp.net/attachments/706368531175964732/706440706624847872/unknown.png" title="mean of each title" alt="mean of each title"></img><br>
다음에서 각 title Age의 평균 값을 df_train['Age']의 Null에 대입해 준다.
```python
df_train.loc[(df_train['Age'].isnull()) & (df_train['Initial'] == 'Mr'), 'Age'] = 33
df_train.loc[(df_train['Age'].isnull()) & (df_train['Initial'] == 'Mrs'), 'Age'] = 36
df_train.loc[(df_train['Age'].isnull()) & (df_train['Initial'] == 'Master'), 'Age'] = 5
df_train.loc[(df_train['Age'].isnull()) & (df_train['Initial'] == 'Miss'), 'Age'] = 22
df_train.loc[(df_train['Age'].isnull()) & (df_train['Initial'] == 'Other'), 'Age'] = 46

df_test.loc[(df_test['Age'].isnull()) & (df_test['Initial'] == 'Mr'), 'Age'] = 33
df_test.loc[(df_test['Age'].isnull()) & (df_test['Initial'] == 'Mrs'), 'Age'] = 36
df_test.loc[(df_test['Age'].isnull()) & (df_test['Initial'] == 'Master'), 'Age'] = 5
df_test.loc[(df_test['Age'].isnull()) & (df_test['Initial'] == 'Miss'), 'Age'] = 22
df_test.loc[(df_test['Age'].isnull()) & (df_test['Initial'] == 'Other'), 'Age'] = 46
```
<br>
<p>train의 Embarked 결측값은 2개로 많지 않기 때문에 최빈값으로 채워준다. <br></p>

```python
df_train['Embarked'].fillna('S', inplace=True)
```
<br>
df_test의 Fare의 Null도 많지 않기 때문에 평균값으로 채워준다.
```python
df_test.loc[df_test.Fare.isnull(), 'Fare'] = df_test['Fare'].mean()
```


#### Categorize Age
Age는 현재 continuous feature이다. 이대로 써도 되지만 category화 시켜줄 수도 있다. 하지만 continuous를 categorical로 바꾸면 information loss가 생길 수 있다. 다만 다양한 방법을 소개하는 것이 목적이기 때문에 categorical하게 바꾸도록 하겠다.
```python
def category_age(x):
    if x < 10:
        return 0
    elif x < 20:
        return 1
    elif x < 30:
        return 2
    elif x < 40:
        return 3
    elif x < 50:
        return 4
    elif x < 60:
        return 5
    elif x < 70:
        return 6
    else:
        return 7    
    
df_train['Age_cat'] = df_train['Age'].apply(category_age)
```
위와 같은 방법으로 바꿔줄 수 있다.

#### String to Numercial (Initial & Embarked & Sex)
String 데이터는 컴퓨터가 인식할 수 있도록 수치화 시켜줘야 한다. 이러한 작업은 map 메소드로 간단히 할 수 있다. Initial과 Embarked, Sex를 수치화 시켜주도록 하겠다.
```python
df_train['Initial'] = df_train['Initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})
df_test['Initial'] = df_test['Initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})
```
```python
df_train['Embarked'] = df_train['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
df_test['Embarked'] = df_test['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
```
```python
df_train['Sex'] = df_train['Sex'].map({'female': 0, 'male': 1})
df_test['Sex'] = df_test['Sex'].map({'female': 0, 'male': 1})
```

#### Skewness
&nbsp;아까 위에서 Fare의 데이터가 왼쪽으로 편향되어있다는 것을 확인했다. 만약 한쪽으로 편향될 경우 outlier에 민감하게 반응해 원하는 모델을 만들지 못할수도 있다고 했다. 이러한 비대칭도는 log를 씌어 잡아줄 수 있다.
```python
df_train['Fare'] = df_train['Fare'].map(lambda i: np.log(i) if i>0 else 0)
df_test['Fare'] = df_test['Fare'].map(lambda i: np.log(i) if i>0 else 0)
```
비대칭도가 잡혔는지 그래프를 그려보면 다음과 같이 잘 잡힌 것을 확인할 수 있다.
```python
f, ax = plt.subplots(1, 1, figsize=(8, 8))
g = sns.distplot(df_train['Fare'], color='b', label='Skewness: {:.2f}'.format(df_train['Fare'].skew()), ax=ax)
g = g.legend(loc='best')
```
<img src="https://cdn.discordapp.com/attachments/706368531175964732/706449082973159504/jTG8bKbiEJkPtCCSiIiIAGopEBEREZeSAhEREQGUFIiIiIhLSYGIiIgASgpERETEpaRAREREACUFIiIi4lJSICIiIoCSAhEREXEp.png" title="Skewness Graph" alt="Skewness Graph"></img><br>

#### Drop Useless Column
```python
df_train.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Age'], axis=1, inplace=True)
df_test.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Age'], axis=1, inplace=True)
```

#### Pearson Correlation
&nbsp;이제 각 feature간의 상관관계를 구하려고 한다. 두 변수간 Pearson Correlation을 구하면 -1 ~ 1 사이의 값을 얻을 수 있다. -1로 갈수록 음의 상관관계, 1로 갈수록 양의 상관관계를 의미하며 0은 상관관계가 없다는 것을 의미한다. Pearson Correlation을 히트맵으로 보면 다음과 같이 나온다.
```python
heatmap_data = df_train[['Survived', 'Pclass', 'Sex', 'Fare', 'Embarked', 'FamilySize', 'Initial', 'Age_cat']]

colormap = plt.cm.BuGn
plt.figure(figsize=(12, 10))
plt.title('Pearson Correalation of Features', y=1.05, size=15)
sns.heatmap(heatmap_data.astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True, annot_kws={'size':16})

plt.show()
```
<img src="https://media.discordapp.net/attachments/706368531175964732/706452177962401802/aqxfMX8j6fA669YZ9ugKeucWGZWIk6gzMzM6ju4zrZKtvxiNwIxM7PiOYEyMzMbnZomJnZ8ziBMjMzq23rogMwM7Pe4wTKzMysho.png?width=762&height=677" title="Pearson Correlation Heatmap" alt="Pearson Correlation Heatmap"></img><br>
위 그림을 보면 우리가 EDA에서 살펴왔듯, Sex와 Pclass가 Survived에 어느정도 상관관계가 있음을 볼 수 있다. 또한 생각보다 Fare와 Embarked도 상관관계가 있음을 볼 수 있다. 그리고 여기서 얻을 수 있는 정보는 서로 강한 상관관계를 가지는 feature가 없다. 만약 서로 다른 두 feature가 1또는 -1의 상관관계를 가진다면 그 두 feature에서 얻을 수 있는 정보는 하나일 것이므로 하나는 지워도 된다.

#### One-hot Encoding (Initial & Embarked)
&nbsp;만약 수치화된 카테고리를 그냥 모델에 넣으면 예를 들어 모델이 Master == 0, Miss == 1, Mr == 2, Mrs == 3, Other == 4와 같은 데이터를 받았을 때, 서로의 호칭들이 상관관계가 있다고 오해한다. 따라서 One-hot Encoding을 해주면 각 데이터가 Master가 맞는지 아닌지, Mrs가 맞는지 아닌지만 판단하기 때문에 오해할 가능성이 적어진다.<br>
&nbsp;One-hot Encoding은 pandas의 get_dummies로 간단하게 할 수 있다.
```python
df_train = pd.get_dummies(df_train, columns=['Initial'], prefix='Initial')
df_test = pd.get_dummies(df_test, columns=['Initial'], prefix='Initial')
```
```python
df_train = pd.get_dummies(df_train, columns=['Embarked'], prefix='Embarked')
df_test = pd.get_dummies(df_test, columns=['Embarked'], prefix='Embarked')
```

### Modeling
&nbsp;길고 긴 여정 끝에 드디어 모델을 만드는 순간까지 왔다. 본 레포지터리는 튜토리얼이기 때문에 모델의 하이퍼 파라미터를 튜닝하진 않겠다.<br>
&nbsp;여기서는 앙상블 모델인 RandomForest를 사용할 것이다. 우선 모듈을 불러올 것이다.
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
```
우리는 킹갓 사이킷런을 사용할 것이다.
&nbsp;우선 데이터셋을 train, test, validation set으로 나눌것이다. train set은 실제 모델을 학습할 때 사용할 데이터이고, test set은 제출할 label이 없는 데이터(여기선 df_test), validation set은 학습시킬 때 사용하지 않고 모델을 임시로 평가할 때 사용한다. 이 데이터셋은 학습에 이용되지 않기 때문에 모델이 얼마나 일반화 됐는지 알 수 있다. 이 데이터셋들은 다음과 같은 코드로 만들어 줄 수 있다.
```python
X_train = df_train.drop('Survived', axis=1).values
target_label = df_train['Survived'].values
X_test = df_test.values
```
```python
X_tr, X_vld, Y_tr, Y_vld = train_test_split(X_train, target_label, test_size=0.3)
```
&nbsp;다음으로 모델을 준비한다. 모델은 사이킷런에서 모두 구현 돼 있기 때문에 우리는 만들어져 있는걸 호출해 오기만 하면 된다. 다음고 ㅏ같은 코드로 호출을 할 수 있다.
```python
model = RandomForestClassifier()
```
그리고 모델 학습은 다음과 같은 코드로 할 수 있다.
```python
model.fit(X_tr, Y_tr)
```

### Evaluating
validation set으로 일반화된 성능은 얼마나 되는지 알 수 있다. 모델 평가하는 방법은 다음과 같다.
```python
# validation set으로 예측
prediction = model.predict(X_vld)
```
```python
print("총 {}명 중 {:.2f}% 정확도로 생존 맞춤".format(Y_vld.shape[0], 100 * metrics.accuracy_score(prediction, Y_vld)))
```
이렇게 예측까지 했다. 나의 경우 정확도는 80.6%가 나왔다.

#### Feature Importance
RandomForest와 같은 트리 기반 모델들은 feature importance를 알 수 있다. feature importance는 다음과 같은 코드로 알 수 있다.
```python
feature_importance = model.feature_importances_
Series_feat_impo = pd.Series(feature_importance, index=df_test.columns)
```
```
plt.figure(figsize=(8, 8))
Series_feat_impo.sort_values(ascending=True).plot.barh()
plt.xlabel('Feature importance')
plt.ylabel('Feature')

plt.show()
```
<img src="https://cdn.discordapp.com/attachments/706368531175964732/706458536103641138/VYiUD6OaMX8DzCHaPV9iejsB9wZIXzZ7v7kcDaRBeEvxDf4VyiP3LREBIrCpkRrvAFdXV3tLoOIiEhHMbMxRKscwH7ufln7SiPSN.png" title="Feature Importance" alt="Feature Importance"></img><br>
이러한 feature importance를 갖고 좀 더 정확한 모델을 위해 feature selection 을 할 수 있고, 좀 더 빠른 모델을 위해 feature를 제거할 수 있다.

### Submission
이제 test set을 예측해 제출해 볼 수 있다. 제출할 때에는 Kaggle에서 주는 gender_submission.csv(왜 이 이름인지는 모르겠지만)에 제출하면 된다.<br>
test를 예측하는 것은 다음과 같이 할 수 있다.
```python
prediction = model.predict(X_test)
```
그리고 제출은 다음과 같이 할 수 있다.
```python
submission = pd.read_csv("gender_submission.csv주소")
submission['Survived'] = prediction
submission.to_csv("/content/gdrive/My Drive/Kaggle Study/Kaggle_Titanic/first_submission.csv", index=False)
```
이렇게 얻은 gender_submission.csv를 Kaggle페이지의 Submit Predictions에 제출하면 된다.
<img src="https://cdn.discordapp.com/attachments/706368531175964732/706461916301623377/unknown.png" title="submission" alt="submission"></img><br>

### 참고자료
http://hero4earth.com/blog/learning/2018/01/29/Feature_Engineering_Basic/ <br>
https://eda-ai-lab.tistory.com/13 <br>
https://statkclee.github.io/model/model-feature-engineering.html <br>
https://www.youtube.com/watch?v=_-N-kdodS0o&list=PLC_wC_PMBL5MnqmgTLqDgu4tO8mrQakuF&index=1 <br>
https://romanegloo.wordpress.com/tag/mr-ms-mrs-miss-mstr-%ED%98%B8%EC%B9%AD/
