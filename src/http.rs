//! A tiny HTTP/1.1 JSON server for the loopback binaries: one thread per connection, every
//! response closes its connection, bodies over 4 MiB refused on the header (413), CORS closed
//! unless the request's Origin is in the allow-list.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::time::Duration;

pub const BODY_LIMIT: usize = 4 * 1024 * 1024;

/// `(method, path, body) → (status, JSON body)`.
pub type Handler = dyn Fn(&str, &str, &[u8]) -> (u16, String) + Send + Sync;

pub fn error_json(msg: &str) -> String {
    serde_json::json!({ "error": msg }).to_string()
}

/// Origins from a comma-separated env var (empty or unset = closed).
pub fn allowed_origins(var: &str) -> Vec<String> {
    std::env::var(var)
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

/// Serve until the process ends.
pub fn serve(bind: &str, allowed: Vec<String>, handler: Arc<Handler>) -> Result<(), String> {
    let listener = TcpListener::bind(bind).map_err(|e| format!("bind {bind}: {e}"))?;
    let allowed = Arc::new(allowed);
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let (handler, allowed) = (handler.clone(), allowed.clone());
        std::thread::spawn(move || {
            let _ = handle(stream, &*handler, &allowed);
        });
    }
    Ok(())
}

fn handle(mut stream: TcpStream, handler: &Handler, allowed: &[String]) -> std::io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(10)))?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();
    let mut len = 0usize;
    let mut origin = None;
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h)? == 0 || h == "\r\n" || h == "\n" {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            match k.trim().to_ascii_lowercase().as_str() {
                "content-length" => len = v.trim().parse().unwrap_or(0),
                "origin" => origin = Some(v.trim().to_string()),
                _ => {}
            }
        }
    }
    let cors = origin.filter(|o| allowed.iter().any(|a| a == o));
    if len > BODY_LIMIT {
        return respond(&mut stream, 413, r#"{"error":"body too large"}"#, cors.as_deref());
    }
    let mut body = vec![0u8; len];
    reader.read_exact(&mut body)?;
    let (status, out) = if method == "OPTIONS" {
        (204, String::new())
    } else {
        handler(&method, &path, &body)
    };
    respond(&mut stream, status, &out, cors.as_deref())
}

fn respond(s: &mut TcpStream, status: u16, body: &str, cors: Option<&str>) -> std::io::Result<()> {
    let text = match status {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        413 => "Payload Too Large",
        422 => "Unprocessable Entity",
        _ => "Error",
    };
    let mut head = format!(
        "HTTP/1.1 {status} {text}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    if let Some(o) = cors {
        head.push_str(&format!(
            "Access-Control-Allow-Origin: {o}\r\nVary: Origin\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type\r\n"
        ));
    }
    head.push_str("\r\n");
    s.write_all(head.as_bytes())?;
    s.write_all(body.as_bytes())?;
    s.flush()
}
