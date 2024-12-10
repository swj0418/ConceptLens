METHODS="sefakmc ganspacekmc"
LAYERS="early_0 early_1 middle_0 middle_1 middle_2 late_0 late_1"
APPLICATION="global layerwise"
DOMAINS="s2_wild512"

### Generation
for domain in ${DOMAINS}
do

for layer in ${LAYERS}
do

for application in ${APPLICATION}
do
for method in ${METHODS}

do
  OUTPUT_FOLDER="cl_server/served_data/${domain}-${method}-${application}-${layer}"
  python process_imageset_enhance.py -r ${OUTPUT_FOLDER}
done
done
done
done
