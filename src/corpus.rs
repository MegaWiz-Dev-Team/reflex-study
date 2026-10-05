//! Corpus directory loader: one `<domain>.md` (one document) or one `<domain>/` folder of
//! `.md` files (one document each) per domain, sorted by name. Non-`.md` files are ignored.
//! A malformed directory is an error with the reason, never a silent fallback.

use std::path::Path;

pub struct DomainDocs {
    pub name: String,
    pub docs: Vec<String>,
}

fn read_doc(p: &Path) -> Result<String, String> {
    let text = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(format!("{} is empty", p.display()));
    }
    Ok(trimmed.to_string())
}

fn is_md(p: &Path) -> bool {
    p.extension().and_then(|e| e.to_str()) == Some("md")
}

pub fn load_dir(dir: &Path) -> Result<Vec<DomainDocs>, String> {
    let mut found: Vec<DomainDocs> = Vec::new();
    let entries = std::fs::read_dir(dir).map_err(|e| format!("cannot read {}: {e}", dir.display()))?;
    for entry in entries {
        let path = entry.map_err(|e| format!("{}: {e}", dir.display()))?.path();
        let Some(name) = path.file_stem().and_then(|s| s.to_str()).map(str::to_string) else {
            continue;
        };
        if name.starts_with('.') {
            continue;
        }
        let docs = if path.is_dir() {
            let mut inner: Vec<_> = std::fs::read_dir(&path)
                .map_err(|e| format!("{}: {e}", path.display()))?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| is_md(p))
                .collect();
            inner.sort();
            if inner.is_empty() {
                return Err(format!("domain `{name}` ({}) has no .md documents", path.display()));
            }
            inner.iter().map(|p| read_doc(p)).collect::<Result<Vec<_>, _>>()?
        } else if is_md(&path) {
            vec![read_doc(&path)?]
        } else {
            continue;
        };
        if found.iter().any(|d| d.name == name) {
            return Err(format!("duplicate domain `{name}`: a `{name}.md` and a `{name}/` both claim it"));
        }
        found.push(DomainDocs { name, docs });
    }
    if found.is_empty() {
        return Err(format!(
            "no domains in {} — expected `<domain>.md` files or `<domain>/` folders of .md files",
            dir.display()
        ));
    }
    found.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(found)
}
