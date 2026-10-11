# python_control 機能ガイド

このドキュメントは、`sample` フォルダの実装をベースに、
本リポジトリの `python_control` 機能の使い方を整理したものです。

- ライブラリ内部アルゴリズムの詳細導出は扱いません
- 「どのクラスをどう組み合わせて使うか」
- 「Pythonで作ったモデルをC++ヘッダへ生成する方法」

を中心に説明します。

## 目次

1. [概要と全体フロー](./01_overview_and_flow.md)
2. [基本クラスの使い方（伝達関数・状態空間・PID）](./02_basic_classes_transfer_state_pid.md)
3. [応用クラスの使い方（LQR/LQI・Kalman・最小二乗）](./03_advanced_classes_lqr_kalman_least_squares.md)
4. [コード生成（Deploy）実践ガイド](./04_cpp_code_generation_guide.md)
5. [可視化と補助ライブラリの役割](./05_visualization_and_external_libraries.md)

## まず読む順番（推奨）

- まずは [概要と全体フロー](./01_overview_and_flow.md)
- すぐにサンプルを動かすなら [基本クラスの使い方（伝達関数・状態空間・PID）](./02_basic_classes_transfer_state_pid.md)
- EKF/UKF やオンライン推定を使うなら [応用クラスの使い方（LQR/LQI・Kalman・最小二乗）](./03_advanced_classes_lqr_kalman_least_squares.md)
- C++実装へ移行するなら [コード生成（Deploy）実践ガイド](./04_cpp_code_generation_guide.md)

