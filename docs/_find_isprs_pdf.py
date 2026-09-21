# -*- coding: utf-8 -*-
"""Find Zotero item WY3KTIRY (or hierarchical semantic graph 2024 ISPRS) and resolve PDF path."""
import json
import shutil
import sqlite3
import tempfile
import urllib.request
from pathlib import Path

ZOTERO_STORAGE = Path(r"C:\Users\Ruiku\Zotero\storage")
DB = Path(r"C:\Users\Ruiku\Zotero\zotero.sqlite")
tmp = Path(tempfile.gettempdir()) / "zotero_ro_wy3.sqlite"
shutil.copy2(DB, tmp)
con = sqlite3.connect(str(tmp))
cur = con.cursor()

# by key
for key in ["WY3KTIRY"]:
    cur.execute(
        """
        SELECT i.itemID, i.key, idv.value
        FROM items i
        JOIN itemData id ON id.itemID=i.itemID AND id.fieldID=(SELECT fieldID FROM fields WHERE fieldName='title')
        JOIN itemDataValues idv ON idv.valueID=id.valueID
        WHERE i.key=?
        """,
        (key,),
    )
    row = cur.fetchone()
    print("BY KEY", key, row)
    if not row:
        continue
    item_id = row[0]
    # attachments
    cur.execute(
        """
        SELECT ia.itemID, ia.key, ia.path, ia.contentType, ia.linkMode,
               COALESCE(idv.value,'') as att_title
        FROM itemAttachments ia
        LEFT JOIN itemData id ON id.itemID=ia.itemID AND id.fieldID=(SELECT fieldID FROM fields WHERE fieldName='title')
        LEFT JOIN itemDataValues idv ON idv.valueID=id.valueID
        WHERE ia.parentItemID=?
        """,
        (item_id,),
    )
    atts = cur.fetchall()
    print("ATTACHMENTS:")
    for a in atts:
        print(a)
        path = a[2] or ""
        if path.startswith("storage:"):
            fname = path.split(":", 1)[1]
            full = ZOTERO_STORAGE / a[1] / fname
            print("  RESOLVED:", full, "exists=", full.exists(), "size=", full.stat().st_size if full.exists() else None)
        elif path.startswith("file:") or Path(path).exists():
            print("  LINKED:", path)

# also search title
cur.execute(
    """
    SELECT i.itemID, i.key, idv.value, COALESCE(idv2.value,'')
    FROM items i
    JOIN itemData id ON id.itemID=i.itemID AND id.fieldID=(SELECT fieldID FROM fields WHERE fieldName='title')
    JOIN itemDataValues idv ON idv.valueID=id.valueID
    LEFT JOIN itemData id2 ON id2.itemID=i.itemID AND id2.fieldID=(SELECT fieldID FROM fields WHERE fieldName='publicationTitle')
    LEFT JOIN itemDataValues idv2 ON idv2.valueID=id2.valueID
    WHERE idv.value LIKE '%hierarchical semantic graph%'
       OR idv.value LIKE '%Hierarchical Semantic Graph%'
       OR (idv.value LIKE '%semantic change%' AND COALESCE(idv2.value,'') LIKE '%ISPRS%')
    LIMIT 20
    """
)
print("\nTITLE SEARCH:")
for r in cur.fetchall():
    print(r)

con.close()

# Also try MCP get_item_details / content
BASE = "http://127.0.0.1:23120/mcp"

def mcp_call(method, params, session=None):
    body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    data = json.dumps(body).encode()
    req = urllib.request.Request(BASE, data=data, headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"})
    if session:
        req.add_header("mcp-session-id", session)
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode()
        sid = resp.headers.get("mcp-session-id")
        return raw, sid

# initialize
raw, sid = mcp_call("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "mimo", "version": "1"}})
print("\nINIT SID", sid)
# notify initialized
try:
    body = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}).encode()
    req = urllib.request.Request(BASE, data=body, headers={"Content-Type": "application/json", "mcp-session-id": sid or ""})
    urllib.request.urlopen(req, timeout=10).read()
except Exception as e:
    print("notify err", e)

raw, _ = mcp_call("tools/call", {"name": "get_item_details", "arguments": {"itemKey": "WY3KTIRY", "mode": "complete"}}, sid)
print("\nMCP DETAILS:\n", raw[:4000])
raw2, _ = mcp_call("tools/call", {"name": "get_content", "arguments": {"itemKey": "WY3KTIRY", "mode": "preview", "format": "text"}}, sid)
print("\nMCP CONTENT preview:\n", raw2[:3000])
