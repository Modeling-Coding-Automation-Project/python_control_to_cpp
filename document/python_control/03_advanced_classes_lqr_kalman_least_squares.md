# 応用クラスの使い方（LQR/LQI・Kalman・最小二乗）

このページでは、以下のサンプルで使われているクラス実装パターンを説明します。

- `sample/lqr/*.py`
- `sample/kalman_filter/*.py`
- `sample/least_squares/*.py`

---

## 1. LQR / LQI

### 対応サンプル

- `sample/lqr/discrete_state_space_lqr.py`
- `sample/lqr/discrete_state_space_lqi.py`
- `sample/lqr/discrete_state_space_lqi_easy.py`

### 実装パターン（LQR）

1. 連続系 `Ac, Bc, Cc, Dc` を定義
2. `c2d` で離散化（`Ad, Bd`）
3. `Q, R` を設定してゲイン `K` を計算
4. 追従ループ内で `u = K(x_ref - x)` を使う
5. DeployでC++定義を生成

```python
from python_control.lqr_deploy import LQR_Deploy
deployed_file_names = LQR_Deploy.generate_LQR_cpp_code(Ac, Bc, Q, R)
```

### 実装パターン（LQI）

LQIは積分器状態を拡張した設計です。

1. `Ac_ex`, `Bc_ex`（拡張系）を構成
2. `Q_ex`, `R_ex` でゲイン `K_ex` を計算
3. `K_x`, `K_e` に分解し、`u = K_x(x_ref - x) + K_e * e_integral`
4. DeployでC++定義を生成

```python
from python_control.lqr_deploy import LQI_Deploy
deployed_file_names = LQI_Deploy.generate_LQI_cpp_code(Ac, Bc, Cc, Q_ex, R_ex)
```

---

## 2. Kalman Filter（LKF / EKF / UKF）

### 対応サンプル

- `sample/kalman_filter/linear_kalman_filter.py`
- `sample/kalman_filter/lkf_fixed_G.py`
- `sample/kalman_filter/extended_kalman_filter.py`
- `sample/kalman_filter/EKF_two_wheel_vehicle_model.py`
- `sample/kalman_filter/unscented_kalman_filter.py`

### 2.1 LKF（線形）

`LinearKalmanFilter(A, B, C, Q, R, delay)` を使い、典型的には以下の順で更新します。

1. 真値系から `y_measured` を得る
2. `predict_and_update(...)` で推定更新
3. `get_x_hat_without_delay()` などで推定値取得

`lkf_fixed_G.py` では事前に `converge_G()` でゲインを収束させ、
`predict_and_update_with_fixed_G(...)` を使う軽量運用例を示しています。

```python
from python_control.kalman_filter_deploy import KalmanFilterDeploy
deployed_file_names = KalmanFilterDeploy.generate_LKF_cpp_code(
    lkf,
    number_of_delay=Number_of_Delay
)
```

### 2.2 EKF（拡張）

EKFサンプルでは、`sympy` から以下を作ります。

- 状態方程式 `f(x,u)`
- 観測方程式 `h(x)`
- それぞれのヤコビアン

そして `ExpressionDeploy.write_*_from_sympy(...)` で関数化したあと、
`ExtendedKalmanFilter(...)` に渡しています。

### 2.3 UKF（無香料）

UKFサンプルでも同様に `f(x,u)` と `h(x)` を関数化し、
`UnscentedKalmanFilter(...)` に渡して推定します。
EKFと違いヤコビアンを明示しない点が実装上の大きな違いです。

```python
deployed_file_names = KalmanFilterDeploy.generate_EKF_cpp_code(ekf, number_of_delay=5)
deployed_file_names = KalmanFilterDeploy.generate_UKF_cpp_code(ukf, number_of_delay=5)
```

---

## 3. 最小二乗（LS / RLS）

### 対応サンプル

- `sample/least_squares/least_squares_method.py`
- `sample/least_squares/least_squares_intercept.py` / `.cpp`（切片の選択とコード生成）
- `sample/least_squares/recursive_squares_method.py`

### 3.1 バッチLS

- `LeastSquares(X)` を生成
- `fit(X, y)` で重み推定
- `predict(X)` / `get_weights()` で結果確認

```python
from external_libraries.MCAP_python_control.python_control.least_squares import LeastSquares
from python_control.least_squares_deploy import LeastSquaresDeploy

ls = LeastSquares(X)
ls.fit(X, y)
deployed_file_names = LeastSquaresDeploy.generate_LS_cpp_code(ls)
```

#### 切片の選択

`fit_intercept` の既定値は `True` で、`y ≈ Xw + b` を推定します。
`LeastSquares(X, fit_intercept=False)` では切片を追加せず、
`min ||Xw - y||²` を解きます。どちらの設定でも、`X` には定数列を
手動で追加しない特徴量行列を渡します。

```python
import numpy as np
from external_libraries.MCAP_python_control.python_control.least_squares import LeastSquares

X = np.array([[1.0], [2.0], [3.0]])
y = np.array([[2.0], [3.0], [4.0]])
with_intercept = LeastSquares(X)
without_intercept = LeastSquares(X, fit_intercept=False)
with_intercept.fit(X, y)
without_intercept.fit(X, y)
print(with_intercept.get_weights().ravel())     # [1. 1.]（係数、切片）
print(without_intercept.get_weights().ravel())  # [1.42857143]（係数のみ）
```

重みは切片ありで「特徴量数＋1」、切片なしで「特徴量数」になり、
切片ありの場合は末尾が切片です。`parameter_count` で係数の数を確認できます。
切片なしでは原点を通るように再学習するため、切片ありの推定結果から
末尾を捨てる方法とは一般に係数が異なります。
`fit_intercept` は読み取り専用です。設定を変える場合は新しいモデルを生成してください。
この選択はバッチLS用であり、RLSは切片を含みます。

比較サンプルはリポジトリルートから次のコマンドで実行できます。
`--no-plot` を省略すると予測結果を可視化します。

```sh
python sample/least_squares/least_squares_intercept.py --no-plot
```

サンプルは切片あり・なしのC++ヘッダーをリポジトリルートに生成し、
ファイル名を表示します。実行時には同名ファイルが上書きされます。

C++では、コンパイル時の第2テンプレート引数に同じ設定を指定します。

```cpp
using X_Type = PythonNumpy::DenseMatrix_Type<double, 20, 2>;
auto with_intercept = PythonControl::make_LeastSquares<X_Type>();
auto without_intercept = PythonControl::make_LeastSquares<X_Type, false>();
// LeastSquares<X_Type, false> / LeastSquares_Type<X_Type, false> も利用可能。
```

`FIT_INTERCEPT`、`PARAMETER_COUNT`、`Weights_Type` で設定と係数次元を参照できます。
`LeastSquaresDeploy` はPythonモデルの設定をC++へ出力します。
生成するファクトリーは未学習モデルを作るため、重みの保存とは別です。
現行C++ソルバーでは「データ行数 > 係数数」が必要です。
切片の追加選択によって、解法・ランク不足への対応は変わりません。

### 3.2 逐次RLS

- `RecursiveLeastSquares(feature_size, lambda_factor)` を生成
- 各サンプルで `update(x, y_true)` を呼ぶ
- 逐次的に重み推定を更新

```python
from external_libraries.MCAP_python_control.python_control.least_squares import RecursiveLeastSquares
from python_control.least_squares_deploy import LeastSquaresDeploy

rls = RecursiveLeastSquares(feature_size=X.shape[1], lambda_factor=0.9)
for i in range(n_samples):
    rls.update(x_i, y_i)
deployed_file_names = LeastSquaresDeploy.generate_RLS_cpp_code(rls)
```

---

次は、各 Deploy クラスの使い分けと生成物（どのヘッダが出るか）をまとめます。
[コード生成（Deploy）実践ガイド](./04_cpp_code_generation_guide.md)

あわせて、C++側の共通API（`solve`, `predict`, `update`, `get_x_hat` など）は
[Pythonクラスに対応するC++クラスとインターフェース](./06_cpp_class_interfaces.md) に整理しています。
