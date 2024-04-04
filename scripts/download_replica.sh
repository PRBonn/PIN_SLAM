echo Creating the dataset path...

mkdir -p data
cd data

# you can also download the Replica.zip manually through
# provided by iMap and Nice-SLAM
echo Downloading Replica dataset...
wget https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip

echo Extracting dataset...
unzip Replica.zip

rm Replica.zip

cd ../..
