### Real-Time EEG Demo Preparation

> CWT Enhancement:  
```
array([ 1.        ,  1.25      ,  1.5625    ,  1.953125  ,  2.44140625,
        3.05175781,  3.81469727,  4.76837158,  5.96046448,  7.4505806 ,
        9.31322575, 11.64153218, 14.55191523, 18.18989404, 22.73736754,
       28.42170943, 35.52713679, 44.40892099, 55.51115123, 69.38893904])
```


### 1. Data Collection & Training
```
1. Get (24, 250 * N) => Get (24, 125 * N)
2. PCA
3. CWT
4. Training
```

### 2. Real-Time Prediction
```
1. Get (24, 250) => Get (24, 125)
2. PCA
3. CWT
4. Predict
```

### 40 mins session
- 80% for training
- 20% for testing