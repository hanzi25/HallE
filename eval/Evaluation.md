# Evaluation

We support:
1. CHAIR
2. POPE
3. MME-Hall
4. AMBER
5. MME

# Evaluation on hallucination benchmarks

## CHAIR

- Validation images from [MSCOCO2014](https://cocodataset.org/#download) and store them at `/raid_sdd/zzy/data/halle/coco/coco2014/val2014`
- We use 500 images for validation

```
##### run chair

bash scripts/v1_5/model_verifier_eval.sh
cd eval/chair
bash eval_chair.sh 
```

## POPE

- Validation images from [MSCOCO2014](https://cocodataset.org/#download) and store them at `/raid_sdd/zzy/data/halle/coco/coco2014/val2014`
- Annotation [data](https://github.com/AoiDragon/POPE/tree/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco) directory and save at `/raid_sdd/zzy/data/pope`

```
#### run pope

cd eval/pope
bash eval.sh 
```

## MME-Hall
- MME-Hall is a subset of MME consisting of `existence`, `count`, `position`, and `color`.
- Follow the official instructions for MME evaluation: [link](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation) and download the MME benchmark. 

```
#### run mme-hall

cd eval/mme
bash eval_mme_hall.sh 
```

## AMBER

- Validation images are from the source repo [AMBER](https://github.com/junyangwang0410/AMBER/tree/master) and store them at `/raid_sdd/whz/data/AMBER/image`. 
- Annotation [data](https://github.com/junyangwang0410/AMBER/tree/master/data) directory and save at `/raid_sdd/whz/data/AMBER/AMBER/data`. 

```
#### run amber

cd eval/amber

# Prepare annotation data (Optional)
python amber_data_prepare.py

# Eval amber
bash eval.sh 
```


# Evaluation on general benchmarks

## MME
- Follow the official instructions for MME evaluation: [link](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation) and download the MME benchmark. 

```
##### run mme

cd eval/mme
bash eval.sh
```