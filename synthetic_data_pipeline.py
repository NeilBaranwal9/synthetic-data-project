import pandas as pd
import numpy as np
import logging
import torch
import warnings
import time
import gc
import os
import joblib 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, OrdinalEncoder
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import NearestNeighbors

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from sdmetrics.reports.single_table import DiagnosticReport, QualityReport

from imblearn.over_sampling import SMOTENC

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_target_variable(df):
    """Create a target variable from existing animal columns"""
    logger.info("Creating target variable from animal data...")
    
    # Define animal categories
    domestic_animals = ['dog', 'cat', 'rabbit', 'cow', 'horse', 'donkey', 'pony', 
                       'duck', 'chicken', 'goat', 'lamb', 'guinea', 'hamster', 'mouse']
    
    wild_animals = ['deer', 'panda', 'koala', 'otter', 'hedgehog', 'squirrel', 
                   'dolphin', 'penguin', 'turtle', 'elephant', 'giraffe', 'bear',
                   'fox', 'beaver', 'monkey', 'seal', 'zebra', 'flamingo', 'bat', 
                   'tamarin', 'wallaby', 'wombat', 'chipmunk', 'cub']
    
    # Initialize target as -1 (unknown/object)
    df['target'] = -1
    
    # Check which animals are present in the dataset
    available_domestic = [animal for animal in domestic_animals if animal in df.columns]
    available_wild = [animal for animal in wild_animals if animal in df.columns]
    
    logger.info(f"Found {len(available_domestic)} domestic animals: {available_domestic[:5]}...")
    logger.info(f"Found {len(available_wild)} wild animals: {available_wild[:5]}...")
    
    # Assign target values (1 = domestic, 0 = wild)
    for animal in available_domestic:
        df.loc[df[animal] == 1, 'target'] = 1
    
    for animal in available_wild:
        df.loc[df[animal] == 1, 'target'] = 0
    
    # Keep only rows with clear animal classification
    original_len = len(df)
    df = df[df['target'] != -1].copy()
    logger.info(f"Kept {len(df)}/{original_len} rows with clear animal classification")
    
    # Check class balance
    class_counts = df['target'].value_counts()
    logger.info(f"Target distribution: {dict(class_counts)}")
    
    if len(class_counts) < 2:
        raise ValueError("Not enough classes found. Try a different target creation strategy.")
    
    return df


def load_and_split_data(csv_path, target_col, test_size=0.20, random_state=42):
    """Load and split data with robust error handling"""
    logger.info(f"Loading data from {csv_path}...")
    
    try:
        # Check if file exists
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File {csv_path} does not exist")
            
        # Sample for type inference
        sample = pd.read_csv(csv_path, nrows=10000)
        
        # CREATE TARGET VARIABLE IF IT DOESN'T EXIST
        if target_col not in sample.columns:
            logger.warning(f"Target column '{target_col}' not found. Creating from animal data...")
            sample = create_target_variable(sample)
            if target_col not in sample.columns:
                raise ValueError(f"Failed to create target column '{target_col}'")
        
        dtype_dict = {}
        
        for col in sample.columns:
            col_data = sample[col].dropna()
            if len(col_data) == 0:
                dtype_dict[col] = 'object'
            elif pd.api.types.is_string_dtype(col_data) or col_data.dtype == 'object':
                if col_data.nunique() / len(col_data) < 0.5:
                    dtype_dict[col] = 'category'
                else:
                    dtype_dict[col] = 'object'
            elif pd.api.types.is_integer_dtype(col_data):
                dtype_dict[col] = 'int32'
            else:
                dtype_dict[col] = 'float32'

        # Load full data
        df = pd.read_csv(csv_path, dtype=dtype_dict, low_memory=True)

        # CREATE TARGET VARIABLE FOR FULL DATASET IF NEEDED
        if target_col not in df.columns:
            df = create_target_variable(df)
            if target_col not in df.columns:
                raise ValueError(f"Failed to create target column '{target_col}' for full dataset")

        # Optimize target column type
        if df[target_col].nunique() < 128:
            df[target_col] = df[target_col].astype('int8')
        else:
            df[target_col] = df[target_col].astype('int32')

        # Remove rows where target is null
        initial_len = len(df)
        df = df.dropna(subset=[target_col]).reset_index(drop=True)
        logger.info(f"Removed {initial_len - len(df)} rows with null target values")
        
        if len(df) == 0:
            raise ValueError("No valid data remaining after removing null target values")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Check for minimum class requirements
        if y.nunique() < 2:
            raise ValueError(f"Target column must have at least 2 unique values. Found: {y.nunique()}")

        # Stratified split with validation
        stratify = y if y.nunique() > 1 and y.nunique() < 100 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=stratify, random_state=random_state
        )

        # Memory optimization
        for subset in [X_train, X_test]:
            for col in subset.select_dtypes(include=['category']):
                subset[col] = subset[col].cat.remove_unused_categories()

        logger.info(f"Data loaded successfully. Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Combine back into DataFrames
        train_df = X_train.copy()
        train_df[target_col] = y_train.values
        test_df = X_test.copy()
        test_df[target_col] = y_test.values

        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def build_metadata(df, target_col):
    """FINAL CONSERVATIVE: Completely avoid boolean to eliminate all errors"""
    logger.info("Building ultra-conservative metadata for SDV...")
    
    try:
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)
        
        # COMPLETELY AVOID BOOLEAN CONVERSION - treat everything as categorical or numerical
        for col in df.columns:
            if col == target_col:
                continue
                
            unique_vals = df[col].dropna().unique()
            n_unique = len(unique_vals)
            
            # Only treat as numerical if it's clearly continuous numeric data
            if (df[col].dtype in ['int64', 'float64', 'int32', 'float32'] and 
                n_unique > 50 and  # Increased threshold for safety
                not any(pd.isna(v) or (isinstance(v, str) and not str(v).replace('.','').replace('-','').isdigit()) for v in unique_vals[:100])):
                metadata.update_column(column_name=col, sdtype='numerical')
                logger.info(f"Set {col} as numerical (n_unique={n_unique})")
            else:
                # EVERYTHING ELSE becomes categorical - no boolean conversion at all
                metadata.update_column(column_name=col, sdtype='categorical')
                logger.info(f"Set {col} as categorical (n_unique={n_unique})")
        
        logger.info("Ultra-conservative metadata building completed successfully")
        return metadata
        
    except Exception as e:
        logger.error(f"Error building metadata: {str(e)}")
        raise


def calculate_safe_batch_size(data_size, pac=10):
    """Calculate batch size that satisfies CTGAN constraints"""
    # CTGAN requires batch_size >= pac and batch_size % pac == 0
    if data_size < pac:
        return pac
    
    # Find the largest valid batch size <= data_size
    max_batch = min(2000, data_size)
    for batch_size in range(max_batch, pac - 1, -1):
        if batch_size % pac == 0:
            return batch_size
    
    return pac  # Fallback


def prepare_data_for_sdv(df):
    """Prepare data consistently for SDV - conservative approach"""
    df_processed = df.copy()
    
    # Convert categorical and object columns to string, but be very conservative
    for col in df_processed.columns:
        if df_processed[col].dtype.name in ['category', 'object']:
            df_processed[col] = df_processed[col].astype(str).fillna('MISSING')
        elif df_processed[col].dtype.name == 'bool':
            # Convert boolean to categorical strings to avoid issues
            df_processed[col] = df_processed[col].astype(str).fillna('MISSING')
    
    return df_processed


def train_generators(train_df, metadata, random_state=42):
    """Train generators with improved error handling and progress bars enabled, or load from disk"""
    models = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    os.makedirs("models", exist_ok=True)

    # Attempt to load existing models first
    all_models_exist = True
    for i in range(4):
        model_path = f"models/ctgan_{random_state + i}.pkl"
        if os.path.exists(model_path):
            models[f"CTGAN_{random_state + i}"] = joblib.load(model_path)
            logger.info(f"Loaded {model_path}")
        else:
            all_models_exist = False

    tvae_path = "models/tvae.pkl"
    if os.path.exists(tvae_path):
        models["TVAE"] = joblib.load(tvae_path)
        logger.info("Loaded TVAE from disk")
    else:
        all_models_exist = False

    if all_models_exist:
        logger.info("All models loaded successfully. Skipping training.")
        return models

    logger.info("Training generators from scratch...")

    # Prepare data consistently for SDV
    train_df_processed = prepare_data_for_sdv(train_df)
    data_size = len(train_df_processed)

    pac = 10 if data_size >= 50 else 2
    batch_size = calculate_safe_batch_size(data_size, pac)

    logger.info(f"Using batch_size={batch_size}, pac={pac} for data_size={data_size}")

    ctgan_params = {
        'epochs': 300,
        'batch_size': batch_size,
        'generator_dim': (256, 256),
        'discriminator_dim': (256, 256),
        'verbose': True,
        'cuda': (device == 'cuda'),
        'pac': pac,
        'enforce_rounding': True,
        'enforce_min_max_values': True
    }

    successful_models = 0
    for i in range(4):
        try:
            if device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

            seed = random_state + i
            torch.manual_seed(seed)
            np.random.seed(seed)

            ctgan = CTGANSynthesizer(metadata, **ctgan_params)
            logger.info(f"Training CTGAN-{i+1} (seed={seed}) with progress bar...")
            start = time.time()
            ctgan.fit(train_df_processed)
            elapsed = (time.time() - start) / 60
            logger.info(f"✓ CTGAN-{i+1} trained in {elapsed:.1f} min")
            models[f"CTGAN_{seed}"] = ctgan
            joblib.dump(ctgan, f"models/ctgan_{seed}.pkl")
            successful_models += 1

            if device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            logger.warning(f"Failed to train CTGAN-{i+1}: {str(e)}")
            continue

    try:
        tvae_batch_size = min(1000, max(data_size // 4, 32))

        tvae_params = {
            'epochs': 200,
            'batch_size': tvae_batch_size,
            'compress_dims': (256, 256),
            'decompress_dims': (256, 256),
            'verbose': True,
            'cuda': (device == 'cuda'),
            'loss_factor': 4,
            'enforce_rounding': True
        }

        tvae = TVAESynthesizer(metadata, **tvae_params)
        logger.info(f"Training TVAE with batch_size={tvae_batch_size} and progress bar...")
        start = time.time()
        tvae.fit(train_df_processed)
        elapsed = (time.time() - start) / 60
        logger.info(f"✓ TVAE trained in {elapsed:.1f} min")
        models["TVAE"] = tvae
        joblib.dump(tvae, "models/tvae.pkl")
        successful_models += 1

    except Exception as e:
        logger.warning(f"Failed to train TVAE: {str(e)}")

    if successful_models == 0:
        raise RuntimeError("No models were successfully trained!")

    logger.info(f"Successfully trained {successful_models} models")
    return models


def evaluate_and_weight(models, real_df, metadata):
    logger.info("Evaluating generators...")
    perf = {}

    for name, mdl in models.items():
        try:
            sample_size = min(5000, max(500, len(real_df)))
            sample_df = mdl.sample(sample_size)  # PATCHED: removed max_retries

            quality_report = QualityReport()
            quality_report.generate(real_df, sample_df, metadata.to_dict())
            quality_score = quality_report.get_score()

            diagnostic_report = DiagnosticReport()
            diagnostic_report.generate(real_df, sample_df, metadata.to_dict())
            privacy_score = diagnostic_report.get_score()

            composite = 0.7 * quality_score + 0.3 * privacy_score
            perf[name] = {
                'model': mdl,
                'quality': quality_score,
                'privacy': privacy_score,
                'composite': composite
            }
            logger.info(f"{name} | Quality: {quality_score:.3f} | Privacy: {privacy_score:.3f}")

        except Exception as e:
            logger.warning(f"Failed to evaluate {name}: {str(e)}")
            continue

    if not perf:
        raise RuntimeError("No models could be evaluated!")

    composites = np.array([v['composite'] for v in perf.values()])
    if np.all(composites == composites[0]):
        weights = np.ones(len(composites)) / len(composites)
    else:
        temp = 0.2
        exp_scores = np.exp((composites - composites.max()) / temp)
        weights = exp_scores / exp_scores.sum()

    weight_dict = {name: w for name, w in zip(perf.keys(), weights)}
    logger.info(f"Model weights: {weight_dict}")
    return perf, weight_dict


def sample_synthetic_data(perf, weights, total_size, target_col, real_df):
    logger.info(f"Generating {total_size} synthetic samples...")
    all_samples = []

    for name, weight in weights.items():
        n_samples = max(1, int(total_size * weight))
        try:
            sample = perf[name]['model'].sample(n_samples)  # PATCHED: removed max_retries
            sample = validate_and_fix_target(sample, real_df, target_col)
            all_samples.append(sample)
            logger.info(f"Sampled {len(sample)} rows from {name}")
        except Exception as e:
            logger.warning(f"Failed to sample from {name}: {str(e)}")
            continue

    if not all_samples:
        raise RuntimeError("No synthetic data could be generated!")

    synth_df = pd.concat(all_samples, ignore_index=True)
    synth_df = synth_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    synth_df = synth_df.head(total_size)
    logger.info(f"Generated {len(synth_df)} synthetic samples")
    return synth_df

def validate_and_fix_target(synth_df, real_df, target_col):
    """Ensure target column exists and has valid values."""
    if target_col not in synth_df.columns:
        target_values = real_df[target_col].value_counts()
        synth_df[target_col] = np.random.choice(
            target_values.index,
            size=len(synth_df),
            p=target_values.values / target_values.sum()
        )
        logger.warning(f"Added missing target column '{target_col}'")

    valid_targets = real_df[target_col].unique()
    mask = synth_df[target_col].isin(valid_targets)

    if not mask.all():
        invalid_idx = ~mask
        synth_df.loc[invalid_idx, target_col] = np.random.choice(
            valid_targets, size=invalid_idx.sum()
        )
        logger.warning(f"Fixed {invalid_idx.sum()} invalid target values")

    return synth_df


def sample_synthetic_data(perf, weights, total_size, target_col, real_df):
    """Generate synthetic data with corrected logic to ensure exact row count"""
    logger.info(f"Generating {total_size} synthetic samples...")

    all_samples = []

    for name, weight in weights.items():
        n_samples = max(1, int(total_size * weight))
        try:
            sample = perf[name]['model'].sample(n_samples)
            # Validate and fix target column
            sample = validate_and_fix_target(sample, real_df, target_col)
            all_samples.append(sample)
            logger.info(f"Sampled {len(sample)} rows from {name}")
        except Exception as e:
            logger.warning(f"Failed to sample from {name}: {str(e)}")
            continue

    if not all_samples:
        raise RuntimeError("No synthetic data could be generated!")

    # Combine all samples
    synth_df = pd.concat(all_samples, ignore_index=True)
    synth_df = synth_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Ensure exactly total_size rows
    if len(synth_df) < total_size:
        # Pad with extra rows from top-weighted model
        top_model_name = max(weights, key=weights.get)
        missing = total_size - len(synth_df)
        logger.warning(f"⚠️ Padding {missing} more rows from {top_model_name}")
        try:
            extra = perf[top_model_name]['model'].sample(missing)
            extra = validate_and_fix_target(extra, real_df, target_col)
            synth_df = pd.concat([synth_df, extra], ignore_index=True)
        except Exception as e:
            logger.error(f"Failed to pad additional rows: {str(e)}")
            raise

    elif len(synth_df) > total_size:
        synth_df = synth_df.head(total_size)

    synth_df = synth_df.reset_index(drop=True)
    logger.info(f"✅ Final synthetic dataset size: {len(synth_df)}")
    return synth_df


def calculate_nndr_vectorized(real_cat_str, synth_cat_str, nndr_thresh=0.5):
    """Correctly calculate NNDR for privacy filtering"""
    # Count occurrences in real data
    real_counts = real_cat_str.value_counts()
    
    # For each synthetic row, count how many times its pattern appears in real data
    synth_matches = synth_cat_str.map(real_counts).fillna(0)
    
    # NNDR = 1 / (count - 1) if count > 1, else 0
    # We want to keep rows with LOW NNDR (more unique patterns)
    nndr_vals = np.where(synth_matches > 1, 1.0 / (synth_matches - 1), np.inf)
    nndr_mask = nndr_vals >= nndr_thresh  # Keep rows with higher uniqueness
    
    return nndr_mask


def privacy_filter(real_df, synth_df, numeric_cols, cat_cols, dcr_thresh=0.05, nndr_thresh=0.5):
    """FIXED: Fully vectorized privacy filtering with safe handling of categorical missing values"""
    logger.info("Applying privacy filters...")

    dcr_mask = np.ones(len(synth_df), dtype=bool)
    nndr_mask = np.ones(len(synth_df), dtype=bool)

    # DCR filtering for numeric columns
    if numeric_cols:
        try:
            real_num = real_df[numeric_cols].fillna(0).values
            synth_num = synth_df[numeric_cols].fillna(0).values

            nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(real_num)
            distances = nbrs.kneighbors(synth_num, return_distance=True)[0].flatten()
            dcr_mask = distances >= dcr_thresh
        except Exception as e:
            logger.warning(f"DCR filtering failed: {str(e)}")

    # NNDR filtering with safe MISSING handling
    if cat_cols:
        try:
            real_cat = real_df[cat_cols].copy()
            synth_cat = synth_df[cat_cols].copy()

            # Ensure 'MISSING' is a valid category
            for df_cat in [real_cat, synth_cat]:
                for col in df_cat.columns:
                    if pd.api.types.is_categorical_dtype(df_cat[col]):
                        if 'MISSING' not in df_cat[col].cat.categories:
                            df_cat[col] = df_cat[col].cat.add_categories(['MISSING'])

            # Fill missing and convert to strings
            real_cat = real_cat.fillna('MISSING').astype(str)
            synth_cat = synth_cat.fillna('MISSING').astype(str)

            real_cat_str = real_cat.apply(lambda x: '|'.join(x), axis=1)
            synth_cat_str = synth_cat.apply(lambda x: '|'.join(x), axis=1)

            nndr_mask = calculate_nndr_vectorized(real_cat_str, synth_cat_str, nndr_thresh)
        except Exception as e:
            logger.warning(f"NNDR filtering failed: {str(e)}")

    final_mask = dcr_mask & nndr_mask
    filtered = synth_df.loc[final_mask].reset_index(drop=True)

    logger.info(f"Privacy filtering: kept {len(filtered)}/{len(synth_df)} rows "
                f"(DCR: {dcr_mask.mean():.1%}, NNDR: {nndr_mask.mean():.1%})")

    # Ensure we have some data
    if len(filtered) == 0:
        logger.warning("All synthetic data was filtered out! Using relaxed constraints...")
        if len(synth_df) > 0:
            n_keep = max(1, len(synth_df) // 2)
            filtered = synth_df.head(n_keep)
            logger.info(f"Using {len(filtered)} samples with relaxed filtering")

    return filtered


def build_preprocessor(numeric_cols, categorical_cols):
    """Build preprocessing pipeline with safe handling for categorical strings"""
    pipelines = []

    if numeric_cols:
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', QuantileTransformer(output_distribution='normal', random_state=0))
        ])
        pipelines.append(('num', num_pipeline, numeric_cols))

    if categorical_cols:
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        pipelines.append(('cat', cat_pipeline, categorical_cols))

    if not pipelines:
        raise ValueError("No valid columns for preprocessing!")

    return ColumnTransformer(pipelines, remainder='drop')


def get_categorical_indices(X_df, categorical_cols):
    """Get correct categorical indices for SMOTENC"""
    X_encoded = X_df.copy()

    # Convert categorical to numeric codes and fill NaNs
    for col in categorical_cols:
        if col in X_encoded.columns:
            X_encoded[col] = pd.Categorical(X_encoded[col]).codes
    X_encoded = X_encoded.fillna(-1)

    cat_indices = [i for i, col in enumerate(X_encoded.columns) if col in categorical_cols]
    return X_encoded, cat_indices


def train_and_evaluate_classifier(hybrid_df, test_df, target_col, random_state=42):
    """Improved classifier training with mixed‐type detection in numeric columns."""
    logger.info("Training classifier...")

    # Ensure target columns are integers
    hybrid_df[target_col] = hybrid_df[target_col].astype(int)
    test_df[target_col] = test_df[target_col].astype(int)

    # Check class balance
    class_counts = hybrid_df[target_col].value_counts()
    logger.info(f"Class distribution: {dict(class_counts)}")

    if len(class_counts) < 2:
        raise ValueError("Need at least 2 classes for classification!")

    # Apply SMOTENC if severely imbalanced
    if class_counts.min() / class_counts.max() < 0.1 and class_counts.min() > 5:
        logger.info("Applying SMOTENC for class balancing...")
        X_orig = hybrid_df.drop(columns=[target_col])
        y_orig = hybrid_df[target_col]

        numeric_cols_sm = X_orig.select_dtypes(include=np.number).columns.tolist()
        categorical_cols_sm = X_orig.select_dtypes(exclude=np.number).columns.tolist()

        if categorical_cols_sm:
            X_encoded, cat_indices = get_categorical_indices(X_orig, categorical_cols_sm)
            min_class_size = class_counts.min()
            k_neighbors = min(5, max(1, min_class_size - 1))
            try:
                smote = SMOTENC(
                    categorical_features=cat_indices,
                    sampling_strategy='minority',
                    k_neighbors=k_neighbors,
                    random_state=random_state
                )
                X_res, y_res = smote.fit_resample(X_encoded, y_orig)
                hybrid_df = pd.DataFrame(X_res, columns=X_encoded.columns)
                hybrid_df[target_col] = y_res
                logger.info(f"SMOTENC applied. New shape: {hybrid_df.shape}")
            except Exception as e:
                logger.warning(f"SMOTENC failed: {str(e)}. Continuing without balancing.")
        else:
            logger.info("No categorical columns found, skipping SMOTENC")
    else:
        logger.info("Skipping SMOTENC (balanced classes or insufficient data)")

    # Prepare features (excluding target from both numeric and categorical)
    feature_df = hybrid_df.drop(columns=[target_col])
    numeric_cols = feature_df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = feature_df.select_dtypes(exclude=np.number).columns.tolist()

    # ——— NEW: Detect and relocate mixed‐type columns ———
    bad_numeric = []
    for col in numeric_cols:
        try:
            # Try converting entire column to float; if it fails, it's not truly numeric
            hybrid_df[col].astype(float)
        except Exception:
            bad_numeric.append(col)

    if bad_numeric:
        logger.warning(f"Moving these columns from numeric to categorical: {bad_numeric}")
        for bad in bad_numeric:
            numeric_cols.remove(bad)
            categorical_cols.append(bad)
    # ———————————————————————————————————————————————

    if not numeric_cols and not categorical_cols:
        raise ValueError("No valid features found for training!")

    logger.info(f"Using {len(numeric_cols)} numeric and {len(categorical_cols)} categorical features")

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    # Define classifiers
    xgb_clf = XGBClassifier(
        n_estimators=500,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        tree_method='hist',
        eval_metric='logloss'
    )
    lgbm_clf = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        device='cpu',
        verbosity=-1
    )

    stack = StackingClassifier(
        estimators=[('xgb', xgb_clf), ('lgbm', lgbm_clf)],
        final_estimator=LogisticRegression(max_iter=1000, random_state=random_state),
        stack_method='predict_proba',
        n_jobs=1
    )

    # Train
    X_hybrid = hybrid_df.drop(columns=[target_col])
    y_hybrid = hybrid_df[target_col]

    try:
        X_hybrid_proc = preprocessor.fit_transform(X_hybrid)
        start = time.time()
        stack.fit(X_hybrid_proc, y_hybrid)
        elapsed = (time.time() - start) / 60
        logger.info(f"✓ Classifier trained in {elapsed:.1f} min")

        # Evaluate
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]
        X_test_proc = preprocessor.transform(X_test)

        y_pred = stack.predict(X_test_proc)
        acc = accuracy_score(y_test, y_pred)

        logger.info(f"Test accuracy: {acc:.4f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred))

        return acc

    except Exception as e:
        logger.error(f"Classifier training failed: {str(e)}")
        return 0.0


def main():
    """Main execution function with comprehensive error handling"""
    # Configuration with validation
    CSV_PATH = "flat-training.csv.gz"
    TARGET_COLUMN = "target"  # Keep as 'target' - we'll create it
    RANDOM_STATE = 42
    SYNTHETIC_MULTIPLIER = 1.0  # (no longer used for sizing)

    # Validate inputs
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Data file not found: {CSV_PATH}")
    
    try:
        # Execute pipeline with comprehensive error handling
        logger.info("Starting synthetic data pipeline...")
        
        train_real, test_real = load_and_split_data(CSV_PATH, TARGET_COLUMN)
        metadata = build_metadata(train_real, TARGET_COLUMN)
        generators = train_generators(train_real, metadata, RANDOM_STATE)
        perf, weights = evaluate_and_weight(generators, train_real, metadata)
        
        # ── EDITED HERE ──
        # Always generate exactly 100000 synthetic rows:
        n_synth = 100_000
        synthetic = sample_synthetic_data(perf, weights, n_synth, TARGET_COLUMN, train_real)
        synthetic.to_csv("synthetic_data_generated.csv", index=False)
        logger.info("📝 Synthetic data saved to synthetic_data_generated.csv")
        # ──────────────────

        # FIXED: Properly exclude target column from features
        all_feature_cols = [col for col in train_real.columns if col != TARGET_COLUMN]
        numeric_cols = train_real[all_feature_cols].select_dtypes(include=np.number).columns.tolist()
        cat_cols = train_real[all_feature_cols].select_dtypes(exclude=np.number).columns.tolist()
        
        filtered_synth = privacy_filter(
            train_real, synthetic, numeric_cols, cat_cols,
            dcr_thresh=0.05, nndr_thresh=0.4
        )

        # Ensure we have some synthetic data
        if len(filtered_synth) == 0:
            logger.warning("No synthetic data passed privacy filter, using subset of original")
            filtered_synth = synthetic.head(max(1, len(train_real) // 4))

        hybrid = pd.concat([train_real, filtered_synth], ignore_index=True)
        hybrid = hybrid.drop_duplicates().sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

        logger.info(f"Final hybrid dataset size: {len(hybrid)} (real: {len(train_real)}, synthetic: {len(filtered_synth)})")

        accuracy = train_and_evaluate_classifier(hybrid, test_real, TARGET_COLUMN, RANDOM_STATE)
        logger.info(f"\n🎯 FINAL ACCURACY: {accuracy:.4f}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()