for d in ./*/* ;
do
	echo "$(ls -l)"
	cd "$d" || exit; # enter each dir if it exists
	echo "entered $d"
	a=1
	for file in *
	do	
		echo "processing $file"
		new_name=$(echo "$d" | sed 's/\.\/downloads\///g')
		echo "$new_name here"
		new=$(printf "%s_%04d.jpg" "$new_name" "$a")
  		#mv "$file" "${file/$d.png}"
		mv "$file" "../$new"
		let a=a+1
	done
	echo "$(ls -l)"
	cd ../..; # back to parent dir
done
