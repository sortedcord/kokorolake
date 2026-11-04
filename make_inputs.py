"""Real intermediate tensors for one sentence (fits L<=64 tokens, F<=128 frames)."""
import sys, numpy as np, torch
from kokoro import KModel, KPipeline
from kparts import *
import dsp
torch.set_grad_enabled(False)
km = bake_weight_norm(KModel(repo_id="hexgrad/Kokoro-82M").eval())
pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=False)
TEXT = sys.argv[1] if len(sys.argv) > 1 else "Okay, the build broke on the main branch."
OUTF = sys.argv[2] if len(sys.argv) > 2 else "bench_inputs.npz"
ph = next(pipe(TEXT, voice="af_aoede")).phonemes
ids = [0, *[km.vocab[p] for p in ph if p in km.vocab], 0]; L = len(ids)
ref_s = pipe.load_voice("af_aoede")[L - 3]
d_en = BertPart(km)(torch.LongTensor([ids]), torch.ones(1, L, dtype=torch.long))
d, dur = ProsodyPart(km)(d_en, ref_s[:, 128:])
pd = np.clip(np.round(dur[0].numpy()), 1, None).astype(np.int64); Fr = int(pd.sum())
aln = np.zeros((L, Fr), np.float32); aln[np.repeat(np.arange(L), pd), np.arange(Fr)] = 1
aln = torch.from_numpy(aln)[None]
f0, n = F0NPart(km)(d.transpose(-1, -2) @ aln, ref_s[:, 128:])
asr = TextEncPart(km)(torch.LongTensor([ids])) @ aln
lin = km.decoder.generator.m_source.l_linear
har = dsp.harmonic_source(f0[0].numpy(), lin.weight[0].numpy(), float(lin.bias[0]), noise=False)
one = lambda k: torch.ones(1, 1, k)
mag, phs = DecoderPart(km)(asr, f0, n, ref_s[:, :128], torch.from_numpy(har)[None], one(Fr), one(2*Fr), one(20*Fr), one(120*Fr+1))
np.savez(OUTF, ids=np.array(ids), d_en=d_en.numpy(), asr=asr.numpy(), f0=f0.numpy(), n=n.numpy(),
         s=ref_s[:, :128].numpy(), har=har, mag=mag.numpy(), phase=phs.numpy())
print(f"L={L} F={Fr} audio={Fr*600/24000:.2f}s")
# self-check: decoder from the saved arrays reproduces the saved output
y = np.load(OUTF)
T_ = lambda k: torch.from_numpy(y[k])
for k in ("asr", "f0", "n", "s", "har"):
    print(k, float(np.abs(y[k] - {"asr": asr, "f0": f0, "n": n, "s": ref_s[:, :128], "har": torch.from_numpy(har)}[k].numpy()).max()))
mag2, _ = DecoderPart(km)(T_("asr"), T_("f0"), T_("n"), T_("s"), T_("har")[None], one(Fr), one(2*Fr), one(20*Fr), one(120*Fr+1))
print("reload check", float((mag2 - mag).abs().max()))
