# Advent Calendar 2021
## 強化学習によるCavity Flow環境（CFD）の境界条件の制御

2DのCavityFlowの非定常計算をCFDで実施し強化学習の環境とする。  
制御目的は制御開始からできるだけ早く、任意の位置の風速を0.2m/s限りなく近づけること。

---
## Setup
下記コマンドでimage build, container runを実行
```sh
cd $PWD/env/rl
docker-compose up -d
docker attach rl_adv_cal_2021
```
実行は下記
```sh
python main.py
```
実行終了したら使わないコンテナは削除しておきましょう。
```sh
cd $PWD/env/rl
docker-compose down
```
---
## 環境(Environment)
2DのCavityFlowの非定常計算をCFDで実施し強化学習の環境とする。環境は下記の計算モデルを採用。Agentのロバスト性を担保する上では環境の初期値等は本来ランダムに設定するべきであるが今回は簡易的な環境に済ませるために初期値は常に同じものとした。

**[barbagroup/CFDPython](https://github.com/barbagroup/CFDPython)**
> [Step 11 —Solves the Navier-Stokes equation for 2D cavity flow.](https://nbviewer.org/github/barbagroup/CFDPython/blob/master/lessons/14_Step_11.ipynb)

### 状態(State: s)
- 代表点(7点)の風速（0~20 m/s）
- TimeStep(0~250)

### 行動(Action: a)
- CFDの境界条件を操作(0.0 ~ 5.0 m/s) -> MAX_VIN

### 報酬(Reward: r)
- 中心座標Cの風速が0.2m/s±0.02m/sで報酬1がもらえる

---

## 強化学習(Reinforcement leraning)
- pytorchベースの[stable_baselines3](https://stable-baselines3.readthedocs.io/en/master/)のライブラリを利用。
- アルゴリズムはPPOかつ環境を2つ用意した並列強化学習。
- 学習のレコード数は62,500
- 評価は5エピソード分実施

---

## 結果の可視化
下記コマンドでimage build, container runを実行
```sh
cd $PWD/env/tensorboard
docker-compose up -d
```
下記URLにアクセスすれば学習結果が確認できます。

SCALERS: http://localhost:6006/#scalars

IMAGES: http://localhost:6006/#images

確認が完了したら使わないコンテナは削除しておきましょう。
```sh
cd $PWD/env/tensorboard
docker-compose down
```