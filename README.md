# Project 512 

## Data 
 - Source Challenges in Representation Learning: Facial Expression Recognition Challenge [line](https://drive.google.com/open?id=1wWiaoJI1A-80V5Rqty5aCDjZaRCczhEN)  
 download data and prepare use:  
  ```bash
	cd ./data
	bash ./download_and_crop.sh 
  ```
## Train 
 ```bash
	cd ./src
	python train.py  
 ```

## Train model 2
 ```bash 
	cd ./src
	python train_mode2.py
 ```	

## Results
  - Ground Truth ./data/fer2013/image
  - tf record ./data/fer2013/tfrecords
  - model and summary ./data/fer2013/train
  - inference ./data/fer2013/inference

## Reference 
 - Use paper from [line]{}
 - read data [line](https://github.com/ZiJiaW/CNN-Face-Expression-Recognition/blob/master/FER.py)
 - CNN training from [line](https://www.tensorflow.org/tutorials/estimators/cnn)


