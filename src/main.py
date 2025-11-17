import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import glob
from matplotlib.colors import Normalize
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt, decimate, correlate
from scipy.ndimage import median_filter


# Assigned ranges for analysis (2 minutes each)
range1 = ("092022", "092212")
range2 = ("091122", "091312")
range3 = ("090302", "090452")


# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

def load_data(ts_start: str, ts_end: str, dx=5.1065, dt=0.0016):
    """Load and concatenate data files for specified time range"""
    files = glob.glob("./data/*.npy")
    files.sort()
    
    def extract_ts(path):
        fname = path.split("\\")[-1].split("/")[-1].split(".")[0]
        return datetime.datetime.strptime("2024-05-07 " + fname, "%Y-%m-%d %H%M%S")
    
    ts_start_dt = datetime.datetime.strptime("2024-05-07 " + ts_start, "%Y-%m-%d %H%M%S")
    ts_end_dt = datetime.datetime.strptime("2024-05-07 " + ts_end, "%Y-%m-%d %H%M%S")
    
    selected = []
    for f in files:
        ts = extract_ts(f)
        if ts_start_dt <= ts <= ts_end_dt:
            selected.append(f)
    
    if not selected:
        raise ValueError("No files found between the given timestamps.")
    
    arrays = [np.load(f) for f in selected]
    data = np.concatenate(arrays)
    
    return data, dt, dx


# ============================================================================
# STEP 2: FREQUENCY ANALYSIS
# ============================================================================

def perform_fft(data, dt):
    """Compute FFT independently for each channel"""
    n_samples, n_channels = data.shape
    freqs = fftfreq(n_samples, dt)
    fft_data = fft(data, axis=0)
    fft_magnitudes = np.abs(fft_data)
    
    return freqs, fft_data, fft_magnitudes


def plot_fft_spectrum(freqs, fft_magnitudes, data, dt):
    """Plot frequency spectrum at single timestamp across channels"""
    # Select middle timestamp
    middle_time_idx = data.shape[0] // 2
    
    # Compute FFT at this timestamp for first 10 channels
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14))
    
    # Top plot: FFT at single timestamp for channels 1-10
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for ch in range(10):
        if ch < data.shape[1]:
            # Get signal at this timestamp for this channel
            signal_snapshot = data[middle_time_idx-500:middle_time_idx+500, ch]  # 1000 samples around middle
            
            # Compute FFT
            fft_ch = np.fft.fft(signal_snapshot)
            fft_ch_mag = np.abs(fft_ch)
            freqs_snapshot = np.fft.fftfreq(len(signal_snapshot), dt)
            
            # Plot only positive frequencies
            positive = freqs_snapshot >= 0
            ax1.semilogy(freqs_snapshot[positive], fft_ch_mag[positive], 
                        linewidth=2, alpha=0.7, color=colors[ch], label=f'Ch {ch+1}')
    
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Magnitude (log scale)')
    ax1.set_title(f'FFT at Time Sample {middle_time_idx} for Channels 1-10')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=1, color='r', linestyle='--', linewidth=2, alpha=0.5)
    ax1.axvline(x=100, color='r', linestyle='--', linewidth=2, alpha=0.5)
    ax1.axvspan(1, 100, alpha=0.1, color='green')
    ax1.legend(ncol=2, fontsize=9)
    ax1.set_xlim([0, 200])
    
    # Middle plot: Spectrogram for channel 1 (frequency changing over time)
    channel_to_analyze = 0
    # Use Short-Time Fourier Transform to show frequency evolution
    window_size = 512
    hop_size = 128
    
    signal = data[:, channel_to_analyze]
    n_windows = (len(signal) - window_size) // hop_size
    
    spectrogram = []
    time_steps = []
    
    for i in range(n_windows):
        start = i * hop_size
        end = start + window_size
        window = signal[start:end]
        
        # Apply FFT to window
        fft_window = np.fft.fft(window)
        fft_mag = np.abs(fft_window[:window_size//2])
        spectrogram.append(fft_mag)
        time_steps.append(start * dt)
    
    spectrogram = np.array(spectrogram).T
    freqs_spec = np.fft.fftfreq(window_size, dt)[:window_size//2]
    
    # Plot spectrogram
    im = ax2.imshow(np.log10(spectrogram + 1e-10), aspect='auto', cmap='hot',
                    extent=[time_steps[0], time_steps[-1], freqs_spec[0], freqs_spec[-1]],
                    origin='lower', interpolation='bilinear')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_title(f'Spectrogram - Channel 1 (Frequency Evolution Over Time)')
    ax2.set_ylim([0, 150])
    ax2.axhline(y=1, color='cyan', linestyle='--', linewidth=1, alpha=0.7)
    ax2.axhline(y=100, color='cyan', linestyle='--', linewidth=1, alpha=0.7)
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('log10(Magnitude)')
    
    # Bottom plot: Temporal FFT for channels 1-10 (full time series)
    positive_freqs = freqs >= 0
    for ch in range(10):
        if ch < fft_magnitudes.shape[1]:
            ax3.semilogy(freqs[positive_freqs], fft_magnitudes[positive_freqs, ch], 
                        linewidth=1.5, alpha=0.7, color=colors[ch], label=f'Ch {ch+1}')
    
    ax3.set_xlabel('Temporal Frequency (Hz)')
    ax3.set_ylabel('Magnitude (log scale)')
    ax3.set_title('Temporal FFT for Channels 1-10 (Full Time Series)')
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=1, color='r', linestyle='--', linewidth=2, label='Lowcut')
    ax3.axvline(x=100, color='r', linestyle='--', linewidth=2, label='Highcut')
    ax3.axvspan(1, 100, alpha=0.1, color='green')
    ax3.legend(ncol=2, fontsize=9)
    ax3.set_xlim([0, 200])
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# STEP 3: NOISE FILTERING AND PREPROCESSING
# ============================================================================

def filter_noise(data, fft_data, freqs, dt, lowcut=1, highcut=100, 
                downsample_factor=2):
    """
    Aggressive multi-stage filtering to clean DAS signal:
    1. Bandpass frequency filtering (lowcut - highcut Hz)
    2. Remove DC and very low frequencies (< 0.5 Hz drift)
    3. Remove common mode noise (spatial filtering)
    4. Remove high-frequency random noise
    5. Spatial coherence filtering
    6. Downsample to reduce data size
    """
    print(f"\nApplying aggressive noise filtering...")
    
    # Stage 1: Bandpass filter in frequency domain
    freq_mask = (np.abs(freqs) >= lowcut) & (np.abs(freqs) <= highcut)
    fft_filtered = fft_data.copy()
    fft_filtered[~freq_mask, :] = 0
    
    # Remove DC and very low frequencies (< 0.5 Hz) - drift/static
    very_low_freq_mask = np.abs(freqs) < 0.5
    fft_filtered[very_low_freq_mask, :] = 0
    
    # Remove very high frequencies (> 80 Hz) - likely noise
    very_high_freq_mask = np.abs(freqs) > 80
    fft_filtered[very_high_freq_mask, :] = 0
    
    # Convert back to time domain
    data_filtered = np.real(np.fft.ifft(fft_filtered, axis=0))
    print(f"  - Bandpass filter: 0.5-80 Hz (was {lowcut}-{highcut})")
    
    # Stage 2: Remove common mode noise (spatial filtering)
    # Average across all channels represents environmental noise
    common_mode = np.mean(data_filtered, axis=1, keepdims=True)
    data_filtered = data_filtered - common_mode
    print(f"  - Common mode noise removed (spatial filtering)")
    
    # Stage 3: Spatial coherence filter
    # Moving objects create coherent patterns across adjacent channels
    # Random noise is incoherent - use median filter in spatial dimension
    from scipy.ndimage import median_filter
    for t in range(0, data_filtered.shape[0], 100):  # Process in chunks
        end_t = min(t + 100, data_filtered.shape[0])
        # Median filter along spatial axis (channels)
        data_filtered[t:end_t, :] = median_filter(data_filtered[t:end_t, :], size=(1, 3), mode='reflect')
    print(f"  - Spatial coherence filter applied (median 3-channel)")
    
    # Stage 4: Remove very weak signals (likely noise)
    # Calculate per-channel threshold
    for ch in range(data_filtered.shape[1]):
        channel_data = data_filtered[:, ch]
        # Keep signals above 10th percentile of absolute values
        threshold = np.percentile(np.abs(channel_data), 10)
        # Soft thresholding: reduce weak signals
        mask = np.abs(channel_data) < threshold
        data_filtered[mask, ch] *= 0.1  # Attenuate weak noise by 90%
    print(f"  - Weak signal attenuation applied")
    
    # Stage 5: Downsample
    if downsample_factor and downsample_factor > 1:
        data_filtered = decimate(data_filtered, downsample_factor, axis=0, ftype='fir')
        dt = dt * downsample_factor
        print(f"  - Downsampled by factor of {downsample_factor}")
    
    print(f"  - Result: std={np.std(data_filtered):.2e}, range=[{np.min(data_filtered):.2e}, {np.max(data_filtered):.2e}]")
    
    return data_filtered, dt


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def set_axis(x, no_labels=7):
    """Helper function for setting plot axis ticks"""
    nx = x.shape[0]
    step_x = int(nx / (no_labels - 1)) 
    x_positions = np.arange(0, nx, step_x) 
    x_labels = x[::step_x]
    return x_positions, x_labels


def get_range(ts_start: str, ts_end: str, dx=5.1065, dt=0.0016):
    """Load data and create DataFrame with proper indexing"""
    files = glob.glob("./data/*.npy")
    files.sort()
    
    def extract_ts(path):
        fname = path.split("\\")[-1].split("/")[-1].split(".")[0]
        return datetime.datetime.strptime("2024-05-07 " + fname, "%Y-%m-%d %H%M%S")
    
    ts_start_dt = datetime.datetime.strptime("2024-05-07 " + ts_start, "%Y-%m-%d %H%M%S")
    ts_end_dt = datetime.datetime.strptime("2024-05-07 " + ts_end, "%Y-%m-%d %H%M%S")
    
    selected = []
    timestamps = []
    for f in files:
        ts = extract_ts(f)
        if ts_start_dt <= ts <= ts_end_dt:
            selected.append(f)
            timestamps.append(ts)
    
    if not selected:
        raise ValueError("No files found between the given timestamps.")
    
    arrays = [np.load(f) for f in selected]
    data = np.concatenate(arrays)
    
    index = pd.date_range(start=timestamps[0], periods=len(data), freq=f"{dt}s")
    columns = np.arange(data.shape[1]) * dx
    df = pd.DataFrame(data=data, index=index, columns=columns)
    
    return {"data": data, "df": df, "dt": dt, "dx": dx}


# ============================================================================
# STEP 4: EXTRACT MOVING OBJECTS
# ============================================================================

def extract_moving_objects(data, dt):
    """
    Remove static background to isolate moving objects
    Uses temporal median filtering to subtract slowly varying background
    """
    print(f"\nExtracting moving objects (background removal)...")
    data_centered = data.copy()
    
    # Calculate appropriate window size: 2-3 seconds for vehicle detection
    # Shorter window = removes static/slow-moving background
    # Preserves fast-moving objects (vehicles)
    window_seconds = 2.5
    window_size = int(window_seconds / dt)
    window_size = window_size if window_size % 2 == 1 else window_size + 1  # Must be odd
    window_size = min(window_size, data.shape[0] // 4)  # Don't exceed 1/4 of data length
    window_size = max(window_size, 11)  # Minimum window size
    
    print(f"  - Temporal median filter: window = {window_size} samples ({window_seconds}s)")
    
    # Apply temporal median filter to each channel
    # This removes static/slowly varying components while preserving transients
    for ch in range(data.shape[1]):
        background = median_filter(data[:, ch], size=window_size, mode='reflect')
        data_centered[:, ch] = data[:, ch] - background
    
    # Enhance contrast by applying soft thresholding
    # Keep only significant deviations from background
    threshold = np.percentile(np.abs(data_centered), 50)  # Median absolute value
    data_centered = np.sign(data_centered) * np.maximum(np.abs(data_centered) - threshold * 0.5, 0)
    print(f"  - Soft thresholding applied (threshold={threshold:.2e})")
    
    # Remove any remaining spatial common mode
    spatial_mean = np.mean(data_centered, axis=1, keepdims=True)
    data_centered = data_centered - spatial_mean
    print(f"  - Spatial common mode removed")
    
    print(f"  - Result: std={np.std(data_centered):.2e}, range=[{np.min(data_centered):.2e}, {np.max(data_centered):.2e}]")
    print(f"  - Non-zero: {np.sum(np.abs(data_centered) > 1e-10)}/{data_centered.size} ({100*np.sum(np.abs(data_centered) > 1e-10)/data_centered.size:.1f}%)")
    
    return data_centered


def plot_processing_stage(data_array, ts_start, ts_end, title, dx=5.1065, dt=0.0016, is_raw=False):
    """Plot data at a specific processing stage using the requested format"""
    # Create DataFrame for proper axis labeling
    pack = get_range(ts_start, ts_end, dx, dt)
    
    # Make sure array size matches
    if data_array.shape[0] > len(pack['df']):
        data_array = data_array[:len(pack['df']), :]
    elif data_array.shape[0] < len(pack['df']):
        # For downsampled data, create appropriate index
        downsample_factor = len(pack['df']) // data_array.shape[0]
        subset_indices = np.arange(0, len(pack['df']), downsample_factor)[:data_array.shape[0]]
        df = pack['df'].iloc[subset_indices]
    else:
        df = pack['df']
    
    # Create DataFrame for axis labeling
    df_vis = pd.DataFrame(data_array.copy(), index=df.index[:data_array.shape[0]], 
                          columns=df.columns[:data_array.shape[1]])
    
    # Only apply mean subtraction and abs to raw data
    # For filtered/processed data, show it as-is to see the actual effect
    if is_raw:
        df_vis = df_vis - df_vis.mean()
        df_vis = np.abs(df_vis)
    else:
        # For processed data, just take absolute value to show amplitude
        df_vis = np.abs(df_vis)
    
    low, high = np.percentile(df_vis, [3, 99])
    norm = Normalize(vmin=low, vmax=high, clip=True)
    
    fig = plt.figure(figsize=(12, 16))
    ax = plt.axes()
    
    im = ax.imshow(df_vis.values, aspect='auto', interpolation='none', norm=norm, cmap='viridis')
    ax.set_ylabel("time")
    ax.set_xlabel("space [m]")
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    cax = fig.add_axes([
        ax.get_position().x1 + 0.06,
        ax.get_position().y0,
        0.02,
        ax.get_position().height
    ])
    plt.colorbar(im, cax=cax)
    
    x_positions, x_labels = set_axis(df_vis.columns)
    ax.set_xticks(x_positions, np.round(x_labels))
    
    y_positions, y_labels = set_axis(df_vis.index.time)
    ax.set_yticks(y_positions, y_labels)
    
    plt.show()
    
    print(f"\n{title}:")
    print(f"  Shape: {data_array.shape}")
    print(f"  Range: [{np.min(data_array):.2e}, {np.max(data_array):.2e}]")
    print(f"  Std: {np.std(data_array):.2e}")
    print(f"  Mean: {np.mean(data_array):.2e}")
    print(f"  Non-zero elements: {np.sum(np.abs(data_array) > 1e-10)}/{data_array.size} "
          f"({100*np.sum(np.abs(data_array) > 1e-10)/data_array.size:.1f}%)")


# ============================================================================
# STEP 5: DETECT OBJECTS AND ESTIMATE VELOCITIES
# ============================================================================

def detect_objects_and_velocities(data, dt, dx, max_lag=1000, debug_plot=False):
    """
    Detect moving objects and estimate velocities using cross-correlation
    between adjacent spatial channels to measure time delay
    """
    n_samples, n_channels = data.shape
    
    print(f"\nDetecting objects using cross-correlation...")
    print(f"  Analyzing {n_channels} channels with {n_samples} samples")
    print(f"  dt={dt:.6f}s, dx={dx:.4f}m")
    
    # Check data quality first
    channel_stds = [np.std(data[:, ch]) for ch in range(min(10, n_channels))]
    print(f"  Signal strength (first 10 channels): {[f'{s:.2e}' for s in channel_stds]}")
    
    if np.max(channel_stds) < 1e-10:
        print("  WARNING: Signal is too weak!")
        return []
    
    # Use cross-correlation between adjacent channels
    # Sample multiple channel pairs across the array
    channel_step = max(1, n_channels // 30)  # Check ~30 pairs
    
    velocities = []
    correlations = []
    
    for i in range(0, n_channels - 1, channel_step):
        signal1 = data[:, i]
        signal2 = data[:, i + 1]
        
        # Check signal strength
        std1, std2 = np.std(signal1), np.std(signal2)
        if std1 < 1e-10 or std2 < 1e-10:
            continue
        
        # Normalize for correlation
        sig1_norm = (signal1 - np.mean(signal1)) / std1
        sig2_norm = (signal2 - np.mean(signal2)) / std2
        
        # Cross-correlation to find time delay
        correlation = correlate(sig2_norm, sig1_norm, mode='same')
        
        # Search for lag around center
        center = len(correlation) // 2
        search_start = max(0, center - max_lag)
        search_end = min(len(correlation), center + max_lag)
        search_region = correlation[search_start:search_end]
        
        # Find peak correlation
        abs_corr = np.abs(search_region)
        max_corr_idx = np.argmax(abs_corr)
        max_corr_value = abs_corr[max_corr_idx]
        
        correlations.append(max_corr_value)
        
        # Lower threshold for weak DAS signals
        correlation_threshold = 0.05
        if max_corr_value > correlation_threshold:
            lag = (search_start + max_corr_idx) - center
            
            if abs(lag) > 3:  # Require minimum lag to avoid false detections
                time_delay = lag * dt
                velocity = dx / abs(time_delay)
                
                # Reasonable velocity range for vehicles (1-50 m/s = 3.6-180 km/h)
                if 1 <= velocity <= 50:
                    velocities.append(velocity)
    
    print(f"  Correlation stats: max={np.max(correlations) if correlations else 0:.3f}, "
          f"mean={np.mean(correlations) if correlations else 0:.3f}")
    print(f"  Found {len(velocities)} velocity measurements")
    
    if len(velocities) == 0:
        print("  No objects detected. Try adjusting filtering parameters.")
        return []
    
    velocities = np.array(velocities)
    print(f"  Velocity range: {np.min(velocities):.1f} - {np.max(velocities):.1f} m/s")
    print(f"               : {np.min(velocities)*3.6:.1f} - {np.max(velocities)*3.6:.1f} km/h")
    
    # Group similar velocities using histogram clustering
    hist, bin_edges = np.histogram(velocities, bins=20)
    detected_objects = []
    threshold = max(1, len(velocities) * 0.1)
    
    for bin_idx in range(len(hist)):
        if hist[bin_idx] >= threshold:
            in_bin = (velocities >= bin_edges[bin_idx]) & (velocities < bin_edges[bin_idx + 1])
            bin_vels = velocities[in_bin]
            
            detected_objects.append({
                'object_id': len(detected_objects) + 1,
                'velocity': np.median(bin_vels),
                'count': hist[bin_idx]
            })
    
    print(f"  Grouped into {len(detected_objects)} distinct objects")
    
    return detected_objects


def plot_correlation_debug(correlations_debug, dt, dx, max_lag):
    """Plot cross-correlation results for debugging"""
    n_pairs = len(correlations_debug)
    fig, axes = plt.subplots(n_pairs, 2, figsize=(14, 4*n_pairs))
    
    if n_pairs == 1:
        axes = axes.reshape(1, -1)
    
    for idx, corr_data in enumerate(correlations_debug):
        ch_pair = corr_data['channel_pair']
        sig1 = corr_data['signal1']
        sig2 = corr_data['signal2']
        correlation = corr_data['correlation']
        search_region = corr_data['search_region']
        max_corr = corr_data['max_corr_value']
        std1 = corr_data['std1']
        std2 = corr_data['std2']
        
        # Plot signals
        time_axis = np.arange(min(500, len(sig1))) * dt
        axes[idx, 0].plot(time_axis, sig1[:500], label=f'Channel {ch_pair[0]}', alpha=0.7)
        axes[idx, 0].plot(time_axis, sig2[:500], label=f'Channel {ch_pair[1]}', alpha=0.7)
        axes[idx, 0].set_title(f'Signals Ch{ch_pair[0]}-Ch{ch_pair[1]} (std={std1:.2e}, {std2:.2e})')
        axes[idx, 0].set_xlabel('Time (s)')
        axes[idx, 0].set_ylabel('Amplitude')
        axes[idx, 0].legend()
        axes[idx, 0].grid(True, alpha=0.3)
        
        # Plot correlation
        center = len(correlation) // 2
        search_start = max(0, center - max_lag)
        lags = np.arange(len(search_region)) - (len(search_region) // 2)
        time_lags = lags * dt
        
        axes[idx, 1].plot(time_lags, search_region)
        axes[idx, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[idx, 1].set_title(f'Cross-correlation (max={max_corr:.3f})')
        axes[idx, 1].set_xlabel('Time Delay (s)')
        axes[idx, 1].set_ylabel('Correlation')
        axes[idx, 1].grid(True, alpha=0.3)
        
        # Mark peak
        peak_idx = np.argmax(np.abs(search_region))
        axes[idx, 1].plot(time_lags[peak_idx], search_region[peak_idx], 'ro', 
                         markersize=10, label=f'Peak at {time_lags[peak_idx]:.4f}s')
        
        # Calculate what velocity this would be
        if time_lags[peak_idx] != 0:
            implied_velocity = dx / abs(time_lags[peak_idx])
            axes[idx, 1].text(0.02, 0.98, f'Velocity: {implied_velocity:.1f} m/s ({implied_velocity*3.6:.1f} km/h)',
                            transform=axes[idx, 1].transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[idx, 1].legend()
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def analyze_segment(ts_start: str, ts_end: str, dx=5.1065, dt=0.0016,
                   lowcut=1, highcut=100, downsample_factor=2, debug_plots=True):
    """
    Complete analysis pipeline:
    1. Load data
    2. Frequency analysis (FFT per channel)
    3. Noise filtering
    4. Extract moving objects
    5. Detect objects and estimate velocities
    """
    
    print(f"\n{'='*60}")
    print(f"Analyzing segment: {ts_start} - {ts_end}")
    print(f"{'='*60}")
    
    # STEP 1: Load data
    print("\nSTEP 1: Loading data...")
    data_raw, dt, dx = load_data(ts_start, ts_end, dx, dt)
    print(f"Loaded data shape: {data_raw.shape}")
    
    # STEP 2: Frequency analysis
    print("\nSTEP 2: Performing FFT (per channel)...")
    freqs, fft_data, fft_mags = perform_fft(data_raw, dt)
    print(f"Frequency range: {np.min(freqs):.2f} to {np.max(freqs):.2f} Hz")
    
    if debug_plots:
        plot_fft_spectrum(freqs, fft_mags, data_raw, dt)
    
    # STEP 3: Noise filtering
    print(f"\nSTEP 3: Filtering noise (bandpass {lowcut}-{highcut} Hz)...")
    data_filtered, dt_filtered = filter_noise(data_raw, fft_data, freqs, dt,
                                              lowcut, highcut, downsample_factor)
    print(f"Filtered data shape: {data_filtered.shape}")
    print(f"New dt after downsampling: {dt_filtered:.6f}s")
    
    # Show FFT after filtering
    if debug_plots:
        print("\nGenerating FFT plots after filtering...")
        freqs_filtered, fft_data_filtered, fft_mags_filtered = perform_fft(data_filtered, dt_filtered)
        plot_fft_spectrum(freqs_filtered, fft_mags_filtered, data_filtered, dt_filtered)
    
    # STEP 4: Extract moving objects
    print("\nSTEP 4: Extracting moving objects...")
    data_moving = extract_moving_objects(data_filtered, dt_filtered)
    
    if debug_plots:
        print("\n--- Visualization of processing stages ---")
        plot_processing_stage(data_raw, ts_start, ts_end, 
                             "STEP 1: Raw Data", dx, dt, is_raw=True)
        plot_processing_stage(data_filtered, ts_start, ts_end, 
                             "STEP 3: After Noise Filtering", dx, dt_filtered, is_raw=False)
        plot_processing_stage(data_moving, ts_start, ts_end, 
                             "STEP 4: Moving Objects (Background Removed)", dx, dt_filtered, is_raw=False)
    
    # STEP 5: Detect objects and estimate velocities
    print("\nSTEP 5: Detecting objects and estimating velocities...")
    detected_objects = detect_objects_and_velocities(data_moving, dt_filtered, dx, 
                                                    debug_plot=debug_plots)
    
    if len(detected_objects) == 0:
        print("\n  Retrying with filtered data (before background removal)...")
        detected_objects = detect_objects_and_velocities(data_filtered, dt_filtered, dx,
                                                        debug_plot=debug_plots)
    
    return {
        'ts_start': ts_start,
        'ts_end': ts_end,
        'data_raw': data_raw,
        'data_filtered': data_filtered,
        'data_moving': data_moving,
        'detected_objects': detected_objects,
        'velocities': [obj['velocity'] for obj in detected_objects],
        'dt': dt,
        'dt_filtered': dt_filtered,
        'dx': dx
    }


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_range(ts_start: str, ts_end: str):
    """Plot raw data for a time range"""
    pack = get_range(ts_start, ts_end)
    df = pack["df"]

    df = df - df.mean()
    df = np.abs(df)

    low, high = np.percentile(df, [3, 99])
    norm = Normalize(vmin=low, vmax=high, clip=True)

    fig = plt.figure(figsize=(12, 16))
    ax = plt.axes()

    im = ax.imshow(df, aspect='auto', interpolation='none', norm=norm)
    ax.set_ylabel("time")
    ax.set_xlabel("space [m]")

    cax = fig.add_axes([
        ax.get_position().x1 + 0.06,
        ax.get_position().y0,
        0.02,
        ax.get_position().height
    ])
    plt.colorbar(im, cax=cax)

    x_positions, x_labels = set_axis(df.columns)
    ax.set_xticks(x_positions, np.round(x_labels))

    y_positions, y_labels = set_axis(df.index.time)
    ax.set_yticks(y_positions, y_labels)

    plt.show()


def plot_results_with_velocities(ts_start, ts_end, detected_objects, dt, dx):
    """Plot original data with velocity lines overlaid"""
    pack = get_range(ts_start, ts_end)
    df = pack["df"]
    
    # Process for visualization
    df_vis = df - df.mean()
    df_vis = np.abs(df_vis)
    
    low, high = np.percentile(df_vis, [3, 99])
    norm = Normalize(vmin=low, vmax=high, clip=True)
    
    n_time, n_space = df_vis.shape
    
    fig, ax = plt.subplots(figsize=(14, 16))
    im = ax.imshow(df_vis.values, aspect='auto', interpolation='none', 
                   norm=norm, cmap='viridis')
    
    if detected_objects:
        for obj in detected_objects:
            velocity = obj['velocity']
            
            if velocity != 0:
                # Calculate slope in pixel coordinates
                slope_pixels = (dx / velocity) / dt
                
                # Draw line across image
                x_pixels = np.array([0, n_space - 1])
                y_pixels = slope_pixels * x_pixels * dx / dt
                
                if y_pixels[1] > 0 and y_pixels[0] < n_time:
                    ax.plot(x_pixels, y_pixels, 'r-', linewidth=2, alpha=0.8)
                    
                    # Add velocity label
                    mid_x = n_space / 2
                    mid_y = slope_pixels * mid_x * dx / dt
                    
                    if 0 <= mid_y < n_time:
                        velocity_kmh = velocity * 3.6
                        ax.text(mid_x, mid_y, f'{velocity_kmh:.1f} km/h', 
                               color='red', fontsize=10, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', 
                                       facecolor='white', alpha=0.7),
                               ha='center', va='bottom')
    
    ax.set_xlabel('space [m]')
    ax.set_ylabel('Time')
    ax.set_title(f'{ts_start} - {ts_end}')
    
    x_positions, x_labels = set_axis(df.columns)
    ax.set_xticks(x_positions, np.round(x_labels))
    
    y_positions, y_labels = set_axis(df.index.time)
    ax.set_yticks(y_positions, y_labels)
    
    cax = fig.add_axes([
        ax.get_position().x1 + 0.02,
        ax.get_position().y0,
        0.02,
        ax.get_position().height
    ])
    plt.colorbar(im, cax=cax, label='Amplitude')
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("DAS Moving Object Detection System")
    print("=" * 60)
    
    # Analyze all three 2-minute segments
    segments = [
        ("Range 1", range1[0], range1[1])
    ]
    
    # segments = [
    #     ("Range 1", range1[0], range1[1]),
    #     ("Range 2", range2[0], range2[1]),
    #     ("Range 3", range3[0], range3[1])
    # ]
    
    results = []
    
    for name, start, end in segments:
        print(f"\n\nProcessing {name}: {start} - {end}")
        result = analyze_segment(start, end, lowcut=1, highcut=100, 
                               downsample_factor=2, debug_plots=True)
        results.append((name, result))
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"SUMMARY FOR {name}")
        print(f"{'='*60}")
        print(f"Detected {len(result['detected_objects'])} moving objects:")
        for obj in result['detected_objects']:
            print(f"  Object {obj['object_id']}: "
                  f"{obj['velocity']:.2f} m/s ({obj['velocity']*3.6:.1f} km/h) "
                  f"[{obj['count']} measurements]")
        
        # Visualize results
        if result['detected_objects']:
            plot_results_with_velocities(start, end, result['detected_objects'], 
                                        result['dt'], result['dx'])
    
    print("\n\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)