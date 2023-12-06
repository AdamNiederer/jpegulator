pub fn write_bmp(bgr: &[u8], width: u32, height: u32) -> Vec<u8> {
    assert!(bgr.len() % 3 == 0);

    let mut ret = Vec::with_capacity(bgr.len() + 12 + 14);

    // BMP header
    ret.extend_from_slice("BM".as_bytes());
    ret.extend_from_slice(&(14 + 40 + bgr.len() as u32 + height * (width % 4)).to_le_bytes());
    ret.extend_from_slice(&0u16.to_le_bytes());
    ret.extend_from_slice(&0u16.to_le_bytes());
    ret.extend_from_slice(&(14u32 + 40u32).to_le_bytes());

    ret.extend_from_slice(&40u32.to_le_bytes());
    ret.extend_from_slice(&width.to_le_bytes());
    ret.extend_from_slice(&height.to_le_bytes());
    ret.extend_from_slice(&1u16.to_le_bytes());
    ret.extend_from_slice(&24u16.to_le_bytes());
    ret.extend_from_slice(&0u32.to_le_bytes());
    ret.extend_from_slice(&0u32.to_le_bytes());
    ret.extend_from_slice(&96u32.to_le_bytes());
    ret.extend_from_slice(&96u32.to_le_bytes());
    ret.extend_from_slice(&0u32.to_le_bytes());
    ret.extend_from_slice(&0u32.to_le_bytes());

    let mut rgb = bgr.to_owned();
    for i in 0..bgr.len() / 3 {
        let tmp = bgr[i * 3];
        rgb[i * 3] = bgr[i * 3 + 2];
        rgb[i * 3 + 2] = tmp;
    }

    for i in (0..height).rev() {
        let start = (i * width * 3) as usize;
        let end = start + (width * 3) as usize;
        ret.extend_from_slice(&rgb[start..end]);
        for _ in 0..(width % 4) {
            ret.push(0x00);
        }
    }

    return ret;
}
