# -*- coding: utf-8 -*-
"""Filter Zotero CD papers relevant to war damage + what 马磊 may like."""
import re
import shutil
import sqlite3
import tempfile
from pathlib import Path

DB = Path(r"C:\Users\Ruiku\Zotero\zotero.sqlite")
tmp = Path(tempfile.gettempdir()) / "zotero_ro_scan2.sqlite"
shutil.copy2(DB, tmp)
con = sqlite3.connect(str(tmp))
cur = con.cursor()

# Get item keys with titles in CD-related collections
sql = """
SELECT DISTINCT
  c.collectionName,
  i.itemID,
  i.key,
  idv.value AS title,
  COALESCE(idv2.value, '') AS date,
  COALESCE(idv3.value, '') AS pub,
  COALESCE(idv4.value, '') AS abstract
FROM collectionItems ci
JOIN collections c ON c.collectionID = ci.collectionID
JOIN items i ON i.itemID = ci.itemID
JOIN itemData id ON id.itemID = i.itemID AND id.fieldID = (SELECT fieldID FROM fields WHERE fieldName='title')
JOIN itemDataValues idv ON idv.valueID = id.valueID
LEFT JOIN itemData id2 ON id2.itemID = i.itemID AND id2.fieldID = (SELECT fieldID FROM fields WHERE fieldName='date')
LEFT JOIN itemDataValues idv2 ON idv2.valueID = id2.valueID
LEFT JOIN itemData id3 ON id3.itemID = i.itemID AND id3.fieldID = (SELECT fieldID FROM fields WHERE fieldName='publicationTitle')
LEFT JOIN itemDataValues idv3 ON idv3.valueID = id3.valueID
LEFT JOIN itemData id4 ON id4.itemID = i.itemID AND id4.fieldID = (SELECT fieldID FROM fields WHERE fieldName='abstractNote')
LEFT JOIN itemDataValues idv4 ON idv4.valueID = id4.valueID
WHERE c.collectionName IN (
  '变化检测','场景变化检测','目标检测','图网络','数据集','综述','地震',
  'OBIA','OBIA in object detection','OBIA in object detection2','背景','小论文','看论文'
)
"""

rows = cur.execute(sql).fetchall()
print(f"raw rows: {len(rows)}")

# Score keywords aligned with 马磊's discussion
themes = {
    "war_damage_conflict": [
        r"war", r"conflict", r"ukraine", r"gaza", r"syria", r"mosul", r"battle",
        r"战争", r"战场", r"冲突", r"加沙", r"乌克兰", r"损毁", r"战损",
    ],
    "building_damage": [
        r"damage", r"damaged", r"collapse", r"earthquake", r"disaster",
        r"xbd", r"xview", r"损毁", r"震害", r"倒塌", r"灾害",
    ],
    "semantic_cd": [
        r"semantic change", r"from-to", r"fromto", r"land cover change",
        r"语义变化", r"场景变化",
    ],
    "graph_label": [
        r"graph neural", r"\bgcn\b", r"\bgat\b", r"hypergraph", r"graph convolution",
        r"图神经", r"图卷积", r"标签空间",
    ],
    "long_tail_imbalance": [
        r"long[- ]tail", r"imbalanc", r"rare class", r"class imbalance",
        r"长尾", r"不平衡", r"稀有",
    ],
    "object_instance": [
        r"object[- ]based", r"object[- ]level", r"instance", r"\bobia\b",
        r"object detection", r"实例", r"面向对象", r"目标级",
    ],
    "special_targets": [
        r"crater", r"trench", r"bunker", r"aircraft", r"vessel", r"ship",
        r"runway", r"airport", r"弹坑", r"战壕", r"掩体", r"机场",
    ],
    "foundation_freq": [
        r"foundation model", r"dinov", r"clip", r"sam\b", r"frequency",
        r"wavelet", r"大模型", r"基础模型", r"频域",
    ],
    "review_survey": [
        r"review", r"survey", r"综述", r"meta-analysis",
    ],
    "scale_cluster": [
        r"multi[- ]scale", r"small object", r"cluster", r"density",
        r"多尺度", r"小目标", r"聚集", r"成群",
    ],
}

hits = {k: [] for k in themes}
for coll, iid, key, title, date, pub, abstract in rows:
    text = f"{title} {abstract}".lower()
    for th, pats in themes.items():
        for p in pats:
            if re.search(p, text, re.I):
                hits[th].append((coll, key, title, date[:10], pub))
                break

for th, items in hits.items():
    # dedupe by title
    seen = set()
    uniq = []
    for it in items:
        t = it[2][:80]
        if t not in seen:
            seen.add(t)
            uniq.append(it)
    print(f"\n{'='*60}\nTHEME {th}: {len(uniq)} unique")
    # prioritize: collection 变化检测/目标检测/场景, journals TGRS/ISPRS/RSE, years 2020+
    def score(it):
        coll, key, title, date, pub = it
        s = 0
        if coll in ("目标检测", "场景变化检测", "图网络", "地震"):
            s += 3
        if coll in ("变化检测",):
            s += 1
        low = (pub + title).lower()
        if "tgrs" in low or "geoscience" in low:
            s += 3
        if "isprs" in low:
            s += 3
        if "remote sensing of environment" in low:
            s += 3
        if "photogramm" in low:
            s += 2
        if "遥感学报" in title or "遥感学报" in pub:
            s += 2
        y = date[:4]
        if y.isdigit() and int(y) >= 2021:
            s += 2
        if y.isdigit() and int(y) >= 2024:
            s += 1
        # 马磊 keywords in title
        tl = title.lower()
        for p in ["damage", "war", "gaza", "ukraine", "crater", "long-tail",
                  "graph", "semantic change", "xbd", "object-based", "imbalance",
                  "hierarchical", "instance"]:
            if p in tl:
                s += 2
        return -s

    uniq.sort(key=score)
    for coll, key, title, date, pub in uniq[:25]:
        print(f"  [{coll}] ({date}) {title[:90]} | {pub[:40]} | key={key}")

con.close()
print("\nDONE")
