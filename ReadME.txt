DATASET downloaded from: https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets
 
Extract these zips and place each of them in ../data/raw/
2025 Q2	ZIP	75.32 MB
2025 Q1	ZIP	121.85 MB
2024 Q4	ZIP	117.24 MB
2024 Q3	ZIP	112.8 MB
2024 Q2	ZIP	113.6 MB
2024 Q1	ZIP	118.58 MB



FOLDER STRUCTURE:
/financial-qa-project/
│
├── data/
│   ├── raw/              # Original SEC files
│   ├── processed/        # Cleaned, structured text chunks
│   └── qa_pairs.json     # 50+ Q/A pairs (used for FT and RAG)
│
├── models/
│   ├── fine_tuned_model/ # Fine-tuned model output
│   └── rag_model/        # RAG embedding/indexing outputs
│
├── notebooks/
│   ├── 1_data_preprocessing.ipynb
│   ├── 2_RAG.ipynb
│   ├── 3_fine_tuning.ipynb
│   └── 4_evaluation.ipynb
│
└── app/
    └── app.py    


streamlit run app/app.py  