fn swap_bgr_rgb(bgr_rgb: &[u8]) -> Vec<u8> {
    assert!(bgr_rgb.len() % 3 == 0);
    let mut rgb_bgr = vec!(0; bgr_rgb.len());
    for i in 0..bgr_rgb.len() / 3 {
        rgb_bgr[i * 3] = bgr_rgb[i * 3 + 2];
        rgb_bgr[i * 3 + 1] = bgr_rgb[i * 3 + 1];
        rgb_bgr[i * 3 + 2] = bgr_rgb[i * 3];
    }
    return rgb_bgr;
}

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

    let rgb = swap_bgr_rgb(bgr);

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

pub fn read_bmp(bmp: &[u8]) -> Vec<u8> {
    assert!(&bmp[0..1] == "BM".as_bytes());

    let width = u16::from_le_bytes((bmp[16], bmp[17]).into()) as usize;
    let height = u16::from_le_bytes((bmp[18], bmp[19]).into()) as usize;
    let mut bgr = Vec::with_capacity(width * height);

    // for i in (0..height).rev() {
    //     let start = (i * width * 3) as usize;
    //     let end = start + (width * 3) as usize;
    //     ret.extend_from_slice(&rgb[start..end]);
    //     for _ in 0..(width % 4) {
    //         ret.push(0x00);
    //     }
    // }

    let rgb = swap_bgr_rgb(&bgr);

    return rgb;
}
