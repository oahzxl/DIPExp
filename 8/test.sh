
model_path=$1

ls demo_images/* | while read fn
do
    echo $fn
    ~/anaconda3/bin/python3 s3_demo.py $fn $model_path
done
