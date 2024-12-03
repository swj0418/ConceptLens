method="sefakmc"
LAYERS="early_0 early_1 middle_0 middle_1 late_0 late_1"
#LAYERS="middle_0"
APPLICATION="global"
DOMAINS="s2_celeba256"
#DOMAINS="s2_ffhq256"

### Generation
for domain in ${DOMAINS}
do

for layer in ${LAYERS}
do

for application in ${APPLICATION}

do
  OUTPUT_FOLDER="output/${domain}-${method}-${application}-${layer}"
  python data_generation/generate_sefa.py --layer ${layer} --application ${application} --domain ${domain} --n_code 200 --edit_dist 5 --seed 10
done
done
done

### Processing
for domain in ${DOMAINS}
do

for layer in ${LAYERS}
do

for application in ${APPLICATION}

do
  OUTPUT_FOLDER="output/${domain}-${method}-${application}-${layer}"
  python process_imageset_enhance.py -r ${OUTPUT_FOLDER}
done
done
done