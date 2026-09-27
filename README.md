# 給湯器 かんたん点検

ADK給湯器の画面に出ている15個の数字を入力して、故障の可能性を確認するオフラインHTML試作である。判定後は、故障の可能性がある場合は修理窓口へ、故障ではない可能性が高い場合は症状に合わせた対処法へ案内する。

## モデルを作り直す

プロジェクトの `data0120.csv` を読み込み、指定された分割・LightGBM設定・検証データからのしきい値決定で、次のファイルを生成する。

```sh
cd /Users/mu-sota/Desktop/LTS/app
/Users/mu-sota/Desktop/LTS/.venv/bin/python build_model.py
```

## 開く

`index.html` をブラウザで開く。外部サイトへの移動はなく、生成された `model.js` と `model_meta.js` を同じフォルダから読み込む。Google Fontsが利用できない場合は、指定した日本語フォントのフォールバックで表示する。

## 写真で数字を読み取る

開始画面の「写真で数字を読み取る」から、給湯器の画面を撮影または選択できる。初回の読み取り時は、CDNからTesseract.jsを読み込むためインターネット接続が必要である。OCRは best effort のため、読み取った数字は確認画面で給湯器の画面と1つずつ必ず照合してから判定する。

OCR用のサンプル画像は、モデルメタデータから次のコマンドで再生成できる。

```sh
cd /Users/mu-sota/Desktop/LTS/app/samples
/Users/mu-sota/Desktop/LTS/.venv/bin/python make_samples.py
```

## テスト

```sh
cd /Users/mu-sota/Desktop/LTS/app
node --test tests/
```

## 今回のモデル指標

モデル生成時の実測値は `build_model.py` の出力と `model_meta.js` の `ADK_META.metrics` に保存される。しきい値は検証データで再現率0.995以上を目標に決定した。実測テスト再現率は検証時の目標と少し異なる場合がある。画面には確率やAUCを表示しない。

今回の測定値は、検証ROC-AUCが0.983489、しきい値が0.032119275553である。生のテスト値では、再現率0.988852、適合率0.514501、訪問削減率0.569000、見逃し件数10である。ユーザー入力を1桁に丸めたテスト値（`X_test.round(1)`）では、再現率0.989967、適合率0.514484、訪問削減率0.568500、見逃し件数9である。
