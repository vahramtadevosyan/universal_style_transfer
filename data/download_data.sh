#!/bin/bash

train_url="http://images.cocodataset.org/zips/train2017.zip"
val_url="http://images.cocodataset.org/zips/val2017.zip"

train_zip="train2017.zip"
val_zip="val2017.zip"

download_and_unzip() {
	local file_url="$1"
        local zip_file="$2"

	wget -O "$zip_file" "$file_url"
	if [ $? -eq 0 ]; then
		echo "Download successful for $zip_file!"
		unzip "$zip_file"
		rm "$zip_file"
	else
		echo "Download failed for $zip_file. Please check the URL and try again."
	fi
}

#download_and_unzip "$train_url" "$train_zip"
download_and_unzip "$val_url" "$val_zip"

