echo Creating the dataset path...

mkdir -p data
cd data

echo Downloading KITTI odometry dataset, sequence 00 subset, first 100 frames ...
wget -O kitti_example.tar.gz -c https://uni-bonn.sciebo.de/s/KwOuBiPZi8vSz2O/download

echo Extracting dataset...
tar -xvf kitti_example.tar.gz

rm kitti_example.tar.gz

cd ../..