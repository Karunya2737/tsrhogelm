printf "NN1,NN2,TrainingTime,RecognitionTime,Accuracy\n">NN_Performance.csv
array1=(500 1000 1500)
array2=(100 250 500)
for i in ${array1[@]}
do
for j in ${array2[@]}
do
python2 -tt ./source/train.py 0 $i $j
python2 -tt ./source/test.py 0
done
done
