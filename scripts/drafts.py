from scripts.ply_to_npy import ply_to_numpy_array
from helpers.readers import read_by_sample, collect_sample_dataframes, plot_multiple_projections_ordered

sample11 = read_by_sample('logs_pcn_20250424_232509', 11)
sample12 = read_by_sample('logs_pcn_20250424_232509', 11)

for data in data_list:
    sample11.append(ply_to_numpy_array(data))

path = '/home/dim26fa/coding/testing/logs_pcn_20250424_232509/'

samples = ['sample11', 'sample12', 'sample13']
samples_good = ['19', '22', '43']

def plot_samples_for_ids(
    log_dir: str,
    sample_numbers: Iterable[int],
    keys: Sequence[str] = ("inputdf", "outputdf"),
    **plot_kwargs,
):
    """Convenience wrapper: read samples, collect dataframes, plot.

    Args:
        log_dir: directory passed to ``read_by_sample``.
        sample_numbers: iterable of sample IDs (integers).
        keys: which entries to extract per sample and their ordering.
        **plot_kwargs: forwarded to ``plot_multiple_projections_ordered``.
    """
    from helpers.readers import read_by_sample

    samples = [read_by_sample(log_dir, num) for num in sample_numbers]
    dataframes = collect_sample_dataframes(samples, keys=keys)
    return plot_multiple_projections_ordered(dataframes, **plot_kwargs)


