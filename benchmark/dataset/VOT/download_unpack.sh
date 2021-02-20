#!/bin/bash

VOT_YEAR="${1:-2018}"
VOT_CHALLENGE="${2:-main}"

if [ "${VOT_CHALLENGE}" == "main" -o "${VOT_CHALLENGE}" == "shortterm" ]; then
    name="VOT${VOT_YEAR}"
else
    name="VOT${VOT_YEAR}_${VOT_CHALLENGE}"
fi


dl="dataset/dl/${name}"

# donwload
mkdir -p "${dl}"
(
    cd "${dl}"
    base_url="http://data.votchallenge.net/vot${VOT_YEAR}/${VOT_CHALLENGE}"
    wget -c "${base_url}/description.json"
    cat description.json | jq -r '.sequences[] | .annotations.url' >annotations.txt
    cat description.json | jq -r '.sequences[] | .channels.color.url' >color.txt
    cat description.json | jq -r '.sequences[] | .name' >list.txt
    mkdir -p annotations
    (
        cd annotations
        cat ../annotations.txt | xargs -P 4 -t -I{} wget -nv -c "${base_url}/{}"
    )
    mkdir -p color
    (
        cd color
        cat ../color.txt | xargs -P 4 -t -I{} wget -nv -c "${base_url}/{}"
    )
)


# unpack
data="dataset/${name}"
echo "${data}"
mkdir -p "${data}"
python "unzip_vot.py" "$dl" "$data"

# create symbolic link
cd ..
ln -sfb "VOT/${data}" "${name}"
