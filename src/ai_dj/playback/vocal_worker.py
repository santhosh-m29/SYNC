"""Cancellable background vocal analysis, isolated from playback CPU/GIL work."""
import multiprocessing as mp
from queue import Empty


def _run(jobs, results, cache_directory):
    import torch
    from ai_dj.analysis.vocals import analyze_track_vocals
    torch.set_num_threads(2)
    while True:
        job = jobs.get()
        if job is None:
            return
        track_id, source = job
        results.put((track_id, analyze_track_vocals(source, cache_directory)))


class VocalWorker:
    def __init__(self, cache_directory):
        context = mp.get_context("spawn")
        self.jobs, self.results = context.Queue(2), context.Queue(2)
        self.process = context.Process(target=_run, args=(self.jobs, self.results, str(cache_directory)),
                                       name="sync-vocals", daemon=True)
        self.process.start()
        self.pending = None

    def submit(self, track):
        self.jobs.put_nowait((track.track_id, track.source_path))
        self.pending = track.track_id

    def poll(self):
        try:
            result = self.results.get_nowait()
        except Empty:
            return None
        self.pending = None
        return result

    def close(self):
        # Terminating an active model job is safe: incomplete stems have no ready
        # marker, and the original music is never modified.
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=2)
        self.jobs.cancel_join_thread()
        self.results.cancel_join_thread()
        self.jobs.close()
        self.results.close()
