"""
script/download_pfam.py

Downloads 150 diverse Pfam families from the UniProt REST API,
processes sequences through the same pipeline as the original dataset,
rebuilds data/processed/proteins.json and data/encoded/*.pt tensors.

Run from project root:
    python script/download_pfam.py

Time: ~20-40 min depending on network.
"""

import os, sys, json, time, random, re
import urllib.request, urllib.parse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

# ── Configuration ─────────────────────────────────────────────────────────────

PROTEINS_JSON   = "data/processed/proteins.json"
ENCODED_DIR     = "data/encoded"
RAW_DIR         = "data/raw"
MAX_LEN         = 700
MIN_LEN         = 50
MAX_PER_FAMILY  = 100
MIN_PER_FAMILY  = 30
PAD_LEN         = 400   # must match CONF["max_len"]
TARGET_FAMILIES = 150
RANDOM_SEED     = 42
REQUEST_DELAY   = 0.6   # seconds between API calls (polite)

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AA_ORDER)}  # 0=PAD
AA_SET    = set(AA_ORDER)

# ── 150 diverse Pfam family accessions ────────────────────────────────────────
# Spans: kinases, proteases, structural, binding, receptors, enzymes,
#        viral, microbial, metabolic, ribosomal, chaperones, immune, plant.

PFAM_FAMILIES = [
    # Kinases / signal
    ("PF00069", "ProteinKinase"),
    ("PF07714", "TyrKinase"),
    ("PF00780", "CheY_Response"),
    ("PF00512", "HisKinase"),
    # Proteases
    ("PF00089", "TrypsinSerine"),
    ("PF00026", "AspartylProtease"),
    ("PF00083", "MetalloProtease"),
    ("PF00112", "CysteinePeptidase"),
    # Structural / scaffold
    ("PF00041", "Fibronectin3"),
    ("PF00047", "Immunoglobulin"),
    ("PF07679", "ImmunoglobulinI"),
    ("PF00008", "EGFLike"),
    ("PF00254", "FKBP"),
    ("PF00023", "Ankyrin"),
    ("PF00646", "FBox"),
    ("PF00400", "WDRepeat"),
    # DNA/RNA binding
    ("PF00010", "HLH"),
    ("PF00096", "ZincFingerC2H2"),
    ("PF13912", "ZincFingerRING"),
    ("PF00595", "PDZ"),
    ("PF00014", "Kunitz"),
    ("PF00018", "SH3"),
    ("PF00017", "SH2"),
    ("PF00076", "RRM_RNA"),
    ("PF00270", "DEAD_Helicase"),
    # GTPases / signaling
    ("PF00071", "Ras_GTPase"),
    ("PF00027", "CyclicNucleotide"),
    ("PF00168", "PH_domain"),
    # Metabolic enzymes
    ("PF00004", "AAA_ATPase"),
    ("PF00005", "ABC_ATPase"),
    ("PF00155", "Aminotransferase"),
    ("PF00175", "Oxidoreductase_NAD"),
    ("PF00107", "AlcoholDehydrogenase"),
    ("PF00118", "GroEL_Chaperonin"),
    ("PF00150", "Cellulase"),
    ("PF00703", "GlycoHydrolase"),
    # Membrane / transport
    ("PF00001", "GPCR_7TM"),
    ("PF00664", "ABC_Membrane"),
    ("PF00230", "MajorIntrinsicProtein"),
    ("PF07690", "MFS_Transporter"),
    ("PF00209", "SodiumSolute"),
    # Oxidoreductases / cofactor
    ("PF00067", "CytochromeP450"),
    ("PF00141", "Peroxidase"),
    ("PF00081", "Ferredoxin"),
    ("PF00115", "CytochromeC"),
    ("PF00173", "CytochromeB"),
    # Ribosomes
    ("PF00411", "RibosomalS11"),
    ("PF00297", "RibosomalL3"),
    ("PF00276", "RibosomalL2"),
    ("PF00466", "RibosomalL10"),
    ("PF00281", "RibosomalL5"),
    ("PF01929", "RibosomalL14"),
    # Chaperones / folding
    ("PF00012", "HSP70"),
    ("PF00226", "DnaJ"),
    ("PF00011", "HspSmall"),
    # Viral
    ("PF00665", "RetroviralIntegrase"),
    ("PF00552", "Integrase_core"),
    # Immune / defense
    ("PF00045", "Hemopexin"),
    ("PF07654", "ImmunoglobulinC"),
    ("PF01582", "TIR_domain"),
    ("PF00619", "CARD"),
    ("PF00079", "Serpin"),
    # Lectins
    ("PF00139", "Legume_lectin"),
    ("PF00059", "CType_lectin"),
    ("PF02368", "Jacalin"),
    # Lipid metabolism
    ("PF00657", "GDSL_Lipase"),
    ("PF01764", "Lipase_3"),
    ("PF00135", "Carboxylesterase"),
    # Nucleases / modification
    ("PF00443", "UCH_Deubiq"),
    ("PF00304", "Gamma_thionin"),
    # Plant
    ("PF00504", "ChlorophyllBind"),
    ("PF00163", "RuBisCO_large"),
    ("PF02696", "RuBisCO_small"),
    ("PF00320", "GATA_zinc"),
    # Bacterial regulators
    ("PF00165", "AraC_HTH"),
    ("PF00126", "MarR_regulator"),
    ("PF00158", "TetR_repressor"),
    # Collagen / ECM
    ("PF01391", "Collagen"),
    ("PF00442", "SPRY"),
    # Other well-studied
    ("PF00070", "Pyridine_Nuc"),
    ("PF00109", "BetaKetoacyl"),
    ("PF00550", "PhospholipidBind"),
    ("PF00777", "Arrestin"),
    ("PF00036", "EF_hand"),
    ("PF01023", "S100_CalciumBind"),
    ("PF00210", "Ferritin"),
    ("PF00313", "ColdShock"),
    ("PF02362", "BetaAmylase"),
    ("PF00125", "CoreHistone"),
    ("PF00538", "LinkerHistone"),
    ("PF00249", "MybDNA"),
    # Ubiquitin system
    ("PF00240", "Ubiquitin"),
    ("PF03107", "RING_Ubox"),
    ("PF00514", "Armadillo"),
    # More domains
    ("PF02985", "HEAT_repeat"),
    ("PF13855", "LRR_repeat"),
    ("PF00560", "LRR_1"),
    ("PF00075", "RNaseH"),
    ("PF00651", "BTB_POZ"),
    ("PF00628", "PHD_finger"),
    ("PF00439", "Bromodomain"),
    ("PF00397", "WW_domain"),
    ("PF00307", "CH_domain"),
    ("PF00612", "IQ_calmodulin"),
    ("PF12796", "Ankyrin_repeat"),
    # Proteolysis
    ("PF02897", "Subtilisin"),
    ("PF00227", "Proteasome_alpha"),
    # More metabolism
    ("PF13561", "ADH_zinc_N"),
    ("PF00128", "AlphaDextrin"),
    ("PF00082", "Peptidase_C1"),
    ("PF00144", "Beta_lactamase"),
    ("PF01535", "PPR_repeat"),
    # Structural proteins
    ("PF00432", "Prenyltransfer"),
    ("PF00042", "Globin"),
    ("PF03764", "EFG_IV"),
    ("PF00177", "RibosomalS7"),
    ("PF00253", "S_layer"),
    ("PF02263", "DsbA_thioredox"),
    ("PF00091", "Tubulin_C"),
    ("PF03953", "Tubulin_binding"),
    ("PF00190", "Cupin_1"),
    ("PF07883", "Cupin_2"),
    ("PF00301", "Rubredoxin"),
    ("PF00108", "Thiolase"),
    ("PF00186", "DHFR"),
    ("PF00171", "Aldedh"),
    ("PF00132", "PrenylCysteine"),
    ("PF01433", "Peptidase_M1"),
    ("PF00151", "Lipase"),
    ("PF02518", "GyraseB_TOPRIM"),
    ("PF00293", "NUDIX"),
    ("PF01612", "DNAseI"),
    ("PF00291", "PALP"),
    ("PF00196", "GntR_regulator"),
    ("PF01776", "RibosomalL28"),
    ("PF00237", "RibosomalL22"),
    ("PF00333", "RibosomalS5"),
    ("PF00181", "RibosomalL2_C"),
    ("PF00344", "SecY_translocon"),
    ("PF00320", "GATA_zinc2"),   # handled by dedup
    ("PF02896", "PotD_permease"),
    ("PF00520", "IonChannel"),
    ("PF01156", "IMP_dehydro"),
    ("PF00392", "GrpE"),
]

# De-duplicate by accession
_seen = set()
_dedup = []
for acc, name in PFAM_FAMILIES:
    if acc not in _seen:
        _seen.add(acc)
        _dedup.append((acc, name))
PFAM_FAMILIES = _dedup[:TARGET_FAMILIES]

# ── utilities ──────────────────────────────────────────────────────────────────

def parse_fasta(text: str):
    records = []
    header, parts = None, []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(parts)))
            header = line[1:]
            parts = []
        else:
            parts.append(line.upper())
    if header is not None:
        records.append((header, "".join(parts)))
    return records


def clean_seqs(records):
    out = []
    for _h, seq in records:
        seq = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', seq)
        if MIN_LEN <= len(seq) <= MAX_LEN:
            out.append(seq)
    return out


def fetch_uniprot_fasta(pfam_acc: str, reviewed_only: bool = True,
                        size: int = 200) -> str:
    reviewed_filter = "+AND+reviewed:true" if reviewed_only else ""
    query = urllib.parse.quote(f"(xref:pfam-{pfam_acc}){reviewed_filter.replace('+', ' ')}")
    url = (f"https://rest.uniprot.org/uniprotkb/search"
           f"?query={query}&format=fasta&size={size}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"[warn] {pfam_acc}: {e}")
        return ""


def encode_sequence(seq: str) -> torch.Tensor:
    tokens = [AA_TO_IDX.get(aa, 0) for aa in seq[:PAD_LEN]]
    tokens += [0] * (PAD_LEN - len(tokens))
    return torch.tensor(tokens, dtype=torch.long)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(RANDOM_SEED)
    os.makedirs(RAW_DIR,                          exist_ok=True)
    os.makedirs(ENCODED_DIR,                      exist_ok=True)
    os.makedirs(os.path.dirname(PROTEINS_JSON),   exist_ok=True)

    # Keep original 21 families
    new_data = {}
    if os.path.exists(PROTEINS_JSON):
        with open(PROTEINS_JSON) as f:
            new_data = json.load(f)
        print(f"Loaded {len(new_data)} existing families from {PROTEINS_JSON}")

    existing_names = set(new_data.keys())
    ok = 0

    print(f"\nFetching {len(PFAM_FAMILIES)} families from UniProt REST API...\n")

    for i, (acc, name) in enumerate(PFAM_FAMILIES):
        tag = f"[{i+1:3d}/{len(PFAM_FAMILIES)}]"

        if name in existing_names:
            print(f"{tag} {name} — already present")
            ok += 1
            continue

        print(f"{tag} {acc} → {name} ...", end=" ", flush=True)

        # Try reviewed first, fall back to all
        text = fetch_uniprot_fasta(acc, reviewed_only=True, size=200)
        records = parse_fasta(text)
        seqs = clean_seqs(records)

        if len(seqs) < MIN_PER_FAMILY:
            text2 = fetch_uniprot_fasta(acc, reviewed_only=False, size=200)
            records2 = parse_fasta(text2)
            seqs = clean_seqs(records2)
            if len(seqs) >= MIN_PER_FAMILY:
                print(f"(unreviewed) ", end="", flush=True)

        if len(seqs) < MIN_PER_FAMILY:
            print(f"only {len(seqs)} clean seqs — skip")
            time.sleep(REQUEST_DELAY)
            continue

        random.shuffle(seqs)
        seqs = seqs[:MAX_PER_FAMILY]

        # Save raw FASTA
        with open(os.path.join(RAW_DIR, f"{name}.fasta"), "w") as f:
            for j, s in enumerate(seqs):
                f.write(f">{acc}_{j}\n{s}\n")

        new_data[name] = seqs
        ok += 1
        print(f"{len(seqs)} seqs")
        time.sleep(REQUEST_DELAY)

    total_seqs = sum(len(v) for v in new_data.values())
    print(f"\n── {ok} families, {total_seqs} total sequences ──\n")

    if len(new_data) < 50:
        print("[abort] Fewer than 50 families — not overwriting proteins.json")
        sys.exit(1)

    with open(PROTEINS_JSON, "w") as f:
        json.dump(new_data, f, indent=2)
    print(f"Saved {PROTEINS_JSON} ({len(new_data)} families)")

    print(f"\nEncoding tensors → {ENCODED_DIR}/")
    for name, seqs in new_data.items():
        tensors = torch.stack([encode_sequence(s) for s in seqs])
        torch.save(tensors, os.path.join(ENCODED_DIR, f"{name}.pt"))
    print(f"  Done — {len(new_data)} .pt files written")

    print("\nAll done. Next steps:")
    print("  python train_protonet.py")
    print("  python script/run_experiments.py")
    print("  python script/kmer_baseline.py")
    print("  python script/blast_baseline.py")
    print("  python script/generate_figures.py")


if __name__ == "__main__":
    main()
