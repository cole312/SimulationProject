import pandas as pd
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel

# 1. Load the simulated signal file
# pandas will automatically skip the lines starting with '#'
csv_filepath = 'outputs/496_cylindersUP_signals_extra_euler_config_1.csv' 
df = pd.read_csv(csv_filepath, comment='#')

# 2. Define the primary axis of your unrotated LTE waveform
# E.g., if LTE0hz.csv has non-zero values in the 1st column, it's X-axis
base_dir = np.array([1.0, 0.0, 0.0]) 

# 3. Process and fit each waveform independently
waveforms = df['file'].unique()

for waveform in waveforms:
    # Subset data for the current waveform
    df_wf = df[df['file'] == waveform]
    
    bvals = df_wf['bval'].values
    signals = df_wf['signal'].values
    
    # Reconstruct the 3x3 rotation matrices for this waveform
    R_matrices = df_wf[['R11', 'R12', 'R13', 
                        'R21', 'R22', 'R23', 
                        'R31', 'R32', 'R33']].values.reshape(-1, 3, 3)
    
    # Calculate bvecs by applying the rotation to the base direction
    # In your simulation: rot_waveform = gradient @ R.T
    bvecs = np.array([base_dir @ R.T for R in R_matrices])
    
    # Normalize vectors to ensure unit length (safeguard against b=0 zeros)
    norms = np.linalg.norm(bvecs, axis=1)
    bvecs = np.divide(bvecs, norms[:, None], out=np.zeros_like(bvecs), where=norms[:, None]!=0)
    
    # 4. Filter for the DTI regime (b <= 2000 s/mm^2)
    # Using b=4500 on a standard Tensor model will heavily distort the FA
    dti_mask = bvals <= 2000
    
    bvals_dti = bvals[dti_mask]
    bvecs_dti = bvecs[dti_mask]
    signals_dti = signals[dti_mask]
    
    # 5. Create the DIPY gradient table
    gtab = gradient_table(bvals=bvals_dti, bvecs=bvecs_dti)
    
    # 6. Fit the Tensor Model
    tenmodel = TensorModel(gtab)
    tenfit = tenmodel.fit(signals_dti)
    
    # Extract metrics
    fa = tenfit.fa
    md = tenfit.md
    
    print(f"Waveform: {waveform}")
    print(f"  -> FA: {fa:.4f}")
    print(f"  -> MD: {md:.4e} mm²/s")
    print("-" * 40)