for d in ./*/ ;
do
        cd "$d" || exit; # enter each dir if it exists
				find . -mindepth 2 -type f -print -exec mv {} . \;  # merge all files 2 levels deep
        find . -type f ! -name '*.jpg' -exec rm '{}' +  # find and rm any non jpg
        ls -d  */ | xargs rm -rf;  # delete any empty dirs
        cd ..; # back to parent dir
done
