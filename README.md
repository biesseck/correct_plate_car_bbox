### 1. MAIN REQUIREMENTS
- python==3.9
- numpy==1.26.4
- scikit-learn==1.4.0
- scipy==1.12.0
- tensorboard==2.15.2
- torch==2.2.0

### 2. CONFIG
```
git clone https://github.com/biesseck/correct_plate_car_bbox.git
cd correct_plate_car_bbox
source ./install.sh
```

### 3. ANALYSE TRACKS
```
python analyse_tracks.py
```

### 4. VIEW CHARTS
```
tensorboard --logdir ./logs_analysis/02122024_105611
http://localhost:6006
```
