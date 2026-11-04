"""Reference: original PyTorch Kokoro (af_aoede) + per-stage CPU timing."""
import time, torch, numpy as np, soundfile as sf
from kokoro import KPipeline, KModel
torch.manual_seed(0)
model = KModel(repo_id="hexgrad/Kokoro-82M").eval()
pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=model)
T = ("Wait, what? All of the tests passed? I honestly didn't think that would work! "
     "Okay... but here's the bad news: the build broke on the main branch, and I'm not sure why. "
     "Do you want me to roll it back, or should I keep digging?")
chunks = []
for r in pipe(T, voice="af_aoede"):
    print(len(r.phonemes), r.phonemes)
    chunks.append(r.audio.numpy())
sf.write("ref/aoede_ref.wav", np.concatenate(chunks), 24000)

# --- per-stage timing on one sentence ---
ph = next(pipe("Okay, but here's the bad news: the build broke on the main branch, and I'm not sure why.", voice="af_aoede")).phonemes
ids = torch.LongTensor([[0, *[model.vocab[p] for p in ph if p in model.vocab], 0]])
ref_s = pipe.load_voice("af_aoede")[ids.shape[1] - 3]  # pack indexed by phoneme count
m = model
T_ = {}
def tic(): return time.perf_counter()
with torch.no_grad():
    for rep in range(3):
        L = ids.shape[1]; lens = torch.tensor([L]); mask = torch.zeros(1, L, dtype=torch.bool)
        t=tic(); bert = m.bert(ids, attention_mask=(~mask).int()); d_en = m.bert_encoder(bert).transpose(-1,-2); T_['bert']=tic()-t
        s = ref_s[:, 128:]
        t=tic(); d = m.predictor.text_encoder(d_en, s, lens, mask); x,_ = m.predictor.lstm(d); dur = torch.sigmoid(m.predictor.duration_proj(x)).sum(-1); T_['dur(lstm)']=tic()-t
        pd = torch.round(dur).clamp(min=1).long().squeeze()
        idx = torch.repeat_interleave(torch.arange(L), pd); aln = torch.zeros(L, idx.shape[0]); aln[idx, torch.arange(idx.shape[0])] = 1; aln=aln[None]
        en = d.transpose(-1,-2) @ aln
        t=tic(); F0, N = m.predictor.F0Ntrain(en, s); T_['F0N(lstm+conv)']=tic()-t
        t=tic(); t_en = m.text_encoder(ids, lens, mask); T_['text_enc(lstm)']=tic()-t
        asr = t_en @ aln
        t=tic(); audio = m.decoder(asr, F0, N, ref_s[:, :128]); T_['decoder+vocoder']=tic()-t
    frames = aln.shape[-1]
    print(f"tokens={L} frames={frames} audio={audio.shape[-1]/24000:.2f}s")
    tot = sum(T_.values())
    for k,v in T_.items(): print(f"  {k:18s} {v*1000:7.1f} ms  {v/tot*100:5.1f}%")
    print(f"  total {tot*1000:.0f} ms (torch threads={torch.get_num_threads()})")
