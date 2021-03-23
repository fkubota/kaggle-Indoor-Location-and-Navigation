<img src='./data/info/readme/001.png' width='1500'>  

# kaggle-Indoor-Location-and-Navigation

[Indoor-Location-and-Navigation](https://www.kaggle.com/c/indoor-location-navigation/overview) コンペのリポジトリ

デバッグ実行: `ipython3 --pdb exp.py`  
docker build: `sh build_docker.sh`  
docker run: `sh run_docker.sh -p 8713 -g 0`  
	- gpu使わない場合は `-g -1` とする

## Links
- [googledrive](https://drive.google.com/drive/u/1/folders/1lFPbS1gHwJabM4CTQju0tiJaJJxHnmEe)
- [issue board](https://github.com/fkubota/kaggle-Indoor-Location-and-Navigation/projects/1)
- [team issue](https://github.com/sinchir0/indoor/issues)
- [永遠の知識](https://experts-j2p4787.slack.com/archives/C01RGU5B7FV)

## Paper
- hoge

## Task
**Description(DeepL)**

スマートフォンは、車で食料品店に行くときも、クリスマスプレゼントを買うときも、どこにでも持ち歩いています。あなたの許可があれば、アプリケーションはあなたの位置情報を使って、状況に応じた情報を提供することができます。例えば、車での行き方を教えてくれたり、お店を探してくれたり、近くのキャンペーン情報を知らせてくれたりします。これらの便利な機能は、GPSによって実現されています。GPSは、最高の精度を得るために屋外での使用が必要です。しかし、ショッピングモールやイベント会場など、大きな建物の中にいることも少なくありません。公共のセンサーとユーザーの許可に基づいた正確な屋内測位により、屋外にいなくても優れた位置情報体験が可能になります。

現在の測位ソリューションは、特に多層階の建物では精度が低く、また小さなデータセットでは一般化が不十分です。さらに、GPSはスマートフォン以前の時代に作られたものです。今日のユースケースでは、屋内では通常得られないような詳細な情報が必要になることがよくあります。

この競技では、屋内測位技術を提供するXYZ10社がMicrosoft Researchと共同で提供するリアルタイムのセンサーデータに基づいて、スマートフォンの屋内での位置を予測することが求められます。ユーザーの協力を得て提供される「アクティブ」なローカリゼーションデータを用いて、デバイスの位置を特定します。レーダーやカメラなどのパッシブなローカリゼーション手法とは異なり、本コンテストで提供されるデータはユーザーの明示的な許可が必要です。このコンペでは、200以上のビルから集められた約3万件の軌跡データを使用します。

成功すれば、製造業、小売業、自律型機器など、幅広い可能性を秘めた研究に貢献できます。より正確な測位が可能になれば、既存の位置情報アプリの改良にもつながります。もしかしたら、次にショッピングモールに行ったときに、あなた自身がその効果を実感できるかもしれません。

謝辞  
XYZ10は、中国の新興屋内測位技術企業です。2017年以降、XYZ10は、約1,000棟の建物から得られたグランドトゥルースを含むWiFi、地磁気、Bluetoothシグネチャのプライバシーに配慮した屋内位置データセットを蓄積している。

Microsoft Researchは、マイクロソフトの研究子会社です。その目的は、学術、政府、産業界の研究者と協力して、技術革新を通じて、最先端のコンピューティングを進め、世界の研究意欲の高い困難な競争問題を解決することです。


## Log

### 20210317
- 0subチームマージ！！

<img src='./data/info/readme/002.png' width='1500'>  


### 20210321
- 今日は[公式リポジトリ](https://github.com/location-competition/indoor-location-competition-20)の確認
  - IMU: 完成軽装装置。加速度計、ジャイロスコープとか計測する。
  - [geoJsonの勉強サイト](https://gis-oer.github.io/gitbook/book/materials/web_gis/GeoJSON/GeoJSON.html)見つけた。
  - type と repolutionの例

    |TYPE|resolution|
    |---|---|
    |Accelerometer | 0.0023956299|
    |Gyroscope |0.0010681152|
    |Magnetometer |0.5996704|
    |AccelerometerUncalibrated |0.0023956299|
    |GyroscopeUncalibratedv |0.0010681152|
    |MagnetometerUncalibrated |0.5996704|

### 20210323
- koukiさんに[wifiデータセット](https://www.kaggle.com/kokitanisaka/indoorunifiedwifids)を落とした