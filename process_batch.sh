METHODS="sefakmc vac svmw ae"
LAYERS="early_0 early_1 middle_0 middle_1 middle_2 late_0 late_1"
APPLICATION="global layerwise"
DOMAINS="s2_ffhq256"

# Count the total number of items to process
total_methods=$(echo ${METHODS} | wc -w)
total_layers=$(echo ${LAYERS} | wc -w)
total_applications=$(echo ${APPLICATION} | wc -w)
total_domains=$(echo ${DOMAINS} | wc -w)

total_items=$((total_methods * total_layers * total_applications * total_domains))

echo "Total number of items to process: ${total_items}"

# Initialize a processed index counter
processed_index=0

### Generation
for application in ${APPLICATION}
do
  for domain in ${DOMAINS}
  do
    for layer in ${LAYERS}
    do
      for method in ${METHODS}
      do
        # Increment and print the processed index
        processed_index=$((processed_index + 1))
        echo "Processing index: ${processed_index} / ${total_items}"

        OUTPUT_FOLDER="cl_server/served_data/${domain}-${method}-${application}-${layer}"
        python process_imageset_enhance.py -r ${OUTPUT_FOLDER}
      done
    done
  done
done
