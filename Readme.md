## Remove Bad/Corrupt Images
To remove corrupt images, run "remove_corrupt_images.py" after changing the "DIRECTORY_PATH" to appropriate directory which contains folders with images.

## Resize Images for faster processing
To resize all the images to a predefined width and height(preset to 250x250, changeable in the code), use "resize_images.py" after changing the "DIRECTORY_PATH" to appropriate directory which contains folders with images. 

## Getting Data ready for "doppelganger.py"
you can use the following script to move all the images out of their folder and renaming them according to the folders they were in.

``` for d in ./*/* ;
do
	echo "$(ls -l)"
	cd "$d" || exit; # enter each dir if it exists
	echo "entered $d"
	a=1
	for file in *
	do	
		echo "processing $file"
		new_name=$(echo "$d" | sed 's/\.\/known\///g')
		echo "$new_name here"
		new=$(printf "%s_%04d.jpg" "$new_name" "$a")
		mv "$file" "../$new"
		let a=a+1
	done
	echo "$(ls -l)"
	cd ../..; # back to parent dir
done
```

## Find Doppelgangers
Run 'doppelganger.py" after changing the following values accordingly

- KNOWN_DIRECTORY_PATH = "path to folder with all the known images"
- UNKNOWN_DIRECTORY_PATH = "path to folder with unknown images"
- EMBEDDING_LIST = "file path for saving/loading embeddings list"
