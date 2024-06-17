use std::collections::HashMap;
use anyhow::anyhow;
use bitvec::prelude::*;
use crate::jpeg::{ValueMCU, RGBMCU, ScanChannel};
use crate::constants::{MCU_ORDER, m1, m2, m3, m4, m5, s0, s1, s2, s3, s4, s5, s6, s7};
use crate::jpeg::{Jpeg, HuffmanTable, FrameChannel, HuffmanTableID, CoefficientMCU};
use crate::bmp::write_bmp;

fn next_sym(bitstream: &BitSlice<u8, Msb0>, table: &HuffmanTable) -> Result<(usize, u8), anyhow::Error> {
    let mut cur_code = 0u16;
    for i in 1..=16 {
        let lsb = if bitstream[i - 1] { 1 } else { 0 };
        cur_code = (cur_code << 1) | lsb;
        if i >= table.min_code_len as usize && i <= table.max_code_len as usize {
            let start = table.offsets[i - table.min_code_len as usize] as usize;
            let end = if i == table.max_code_len as usize { table.codes.len() } else { table.offsets[1 + i - table.min_code_len as usize] as usize };
            if let Some(pos) = table.codes[start..end].iter().position(|sym| *sym == cur_code) {
                return Ok((i, table.symbols[start + pos]))
            }
        }
    }
    return Err(anyhow!("invalid bitstream: {:?}", &bitstream[0..16]));
}

fn next_coeff(bitstream: &BitSlice<u8, Msb0>, len: usize) -> i16 {
    let mut cur_sym = 0i16;
    for i in 0..len {
        let lsb = if bitstream[i] { 1 } else { 0 };
        cur_sym = (cur_sym << 1) | lsb;
    }
    return cur_sym
}

fn normalize_coeff(coeff: i16, len: usize) -> i16 {
    if len > 0 && coeff < (1 << (len - 1)) {
        coeff - (1 << len) + 1
    } else {
        coeff
    }
}

pub(crate) fn inverse_dct(
    mcu: &CoefficientMCU,
    channels: &HashMap<u8, FrameChannel>,
    channel_order: &Vec<u8>
) -> Result<ValueMCU, anyhow::Error> {
    let mut ret = HashMap::new();

    for channel_id in channel_order {
        let mcu_data = mcu.get(&channel_id).ok_or(anyhow!("no channel {}", channel_id))?;
        let vsf = channels.get(&channel_id).unwrap().vertical_sampling_factor as usize;
        let hsf = channels.get(&channel_id).unwrap().horizontal_sampling_factor as usize;
        let mut out = vec!(0.; 64 * vsf * hsf);

        for block in 0..(hsf * vsf) {
            for i in 0..8 {
                let g0 = mcu_data[64 * block + 0 * 8 + i] as f32 * s0;
                let g1 = mcu_data[64 * block + 4 * 8 + i] as f32 * s4;
                let g2 = mcu_data[64 * block + 2 * 8 + i] as f32 * s2;
                let g3 = mcu_data[64 * block + 6 * 8 + i] as f32 * s6;
                let g4 = mcu_data[64 * block + 5 * 8 + i] as f32 * s5;
                let g5 = mcu_data[64 * block + 1 * 8 + i] as f32 * s1;
                let g6 = mcu_data[64 * block + 7 * 8 + i] as f32 * s7;
                let g7 = mcu_data[64 * block + 3 * 8 + i] as f32 * s3;

                let f0 = g0;
                let f1 = g1;
                let f2 = g2;
                let f3 = g3;
                let f4 = g4 - g7;
                let f5 = g5 + g6;
                let f6 = g5 - g6;
                let f7 = g4 + g7;

                let e0 = f0;
                let e1 = f1;
                let e2 = f2 - f3;
                let e3 = f2 + f3;
                let e4 = f4;
                let e5 = f5 - f7;
                let e6 = f6;
                let e7 = f5 + f7;
                let e8 = f4 + f6;

                let d0 = e0;
                let d1 = e1;
                let d2 = e2 * m1;
                let d3 = e3;
                let d4 = e4 * m2;
                let d5 = e5 * m3;
                let d6 = e6 * m4;
                let d7 = e7;
                let d8 = e8 * m5;

                let c0 = d0 + d1;
                let c1 = d0 - d1;
                let c2 = d2 - d3;
                let c3 = d3;
                let c4 = d4 + d8;
                let c5 = d5 + d7;
                let c6 = d6 - d8;
                let c7 = d7;
                let c8 = c5 - c6;

                let b0 = c0 + c3;
                let b1 = c1 + c2;
                let b2 = c1 - c2;
                let b3 = c0 - c3;
                let b4 = c4 - c8;
                let b5 = c8;
                let b6 = c6 - c7;
                let b7 = c7;

                out[64 * block + 0 * 8 + i] = b0 + b7;
                out[64 * block + 1 * 8 + i] = b1 + b6;
                out[64 * block + 2 * 8 + i] = b2 + b5;
                out[64 * block + 3 * 8 + i] = b3 + b4;
                out[64 * block + 4 * 8 + i] = b3 - b4;
                out[64 * block + 5 * 8 + i] = b2 - b5;
                out[64 * block + 6 * 8 + i] = b1 - b6;
                out[64 * block + 7 * 8 + i] = b0 - b7;
            }

            for i in 0..8 {
                let g0 = out[64 * block + i * 8 + 0] * s0;
                let g1 = out[64 * block + i * 8 + 4] * s4;
                let g2 = out[64 * block + i * 8 + 2] * s2;
                let g3 = out[64 * block + i * 8 + 6] * s6;
                let g4 = out[64 * block + i * 8 + 5] * s5;
                let g5 = out[64 * block + i * 8 + 1] * s1;
                let g6 = out[64 * block + i * 8 + 7] * s7;
                let g7 = out[64 * block + i * 8 + 3] * s3;

                let f0 = g0;
                let f1 = g1;
                let f2 = g2;
                let f3 = g3;
                let f4 = g4 - g7;
                let f5 = g5 + g6;
                let f6 = g5 - g6;
                let f7 = g4 + g7;

                let e0 = f0;
                let e1 = f1;
                let e2 = f2 - f3;
                let e3 = f2 + f3;
                let e4 = f4;
                let e5 = f5 - f7;
                let e6 = f6;
                let e7 = f5 + f7;
                let e8 = f4 + f6;

                let d0 = e0;
                let d1 = e1;
                let d2 = e2 * m1;
                let d3 = e3;
                let d4 = e4 * m2;
                let d5 = e5 * m3;
                let d6 = e6 * m4;
                let d7 = e7;
                let d8 = e8 * m5;

                let c0 = d0 + d1;
                let c1 = d0 - d1;
                let c2 = d2 - d3;
                let c3 = d3;
                let c4 = d4 + d8;
                let c5 = d5 + d7;
                let c6 = d6 - d8;
                let c7 = d7;
                let c8 = c5 - c6;

                let b0 = c0 + c3;
                let b1 = c1 + c2;
                let b2 = c1 - c2;
                let b3 = c0 - c3;
                let b4 = c4 - c8;
                let b5 = c8;
                let b6 = c6 - c7;
                let b7 = c7;

                out[64 * block + i * 8 + 0] = b0 + b7 + 0.5;
                out[64 * block + i * 8 + 1] = b1 + b6 + 0.5;
                out[64 * block + i * 8 + 2] = b2 + b5 + 0.5;
                out[64 * block + i * 8 + 3] = b3 + b4 + 0.5;
                out[64 * block + i * 8 + 4] = b3 - b4 + 0.5;
                out[64 * block + i * 8 + 5] = b2 - b5 + 0.5;
                out[64 * block + i * 8 + 6] = b1 - b6 + 0.5;
                out[64 * block + i * 8 + 7] = b0 - b7 + 0.5;
            }
        }

        ret.insert(*channel_id, out.iter().map(|v| *v as i8).collect::<Vec<i8>>());
    }
    Ok(ret)
}

pub(crate) fn dequantize<'a>(
    mcu: &'a mut CoefficientMCU,
    quant_tables: &HashMap<usize, Vec<u16>>,
    channels: &HashMap<u8, FrameChannel>
) -> Result<&'a CoefficientMCU, anyhow::Error> {
    for channel_id in channels.keys() {
        let channel = channels.get(channel_id).ok_or(anyhow!("no channel {}", channel_id))?;

        let mcu_data = mcu.get_mut(channel_id).ok_or(anyhow!("no mcu for {}", channel_id))?;
        let quant_data = quant_tables.get(&channel.quant_table_id).ok_or(anyhow!("no quant for {}", channel.quant_table_id))?;
        for i in 0..mcu_data.len() {
            mcu_data[i] *= quant_data[i % 64] as i16;
        }
    }

    Ok(mcu)
}

fn block_decode(
    bitstream: &BitSlice<u8, Msb0>,
    dc_table: &HuffmanTable,
    ac_table: &HuffmanTable,
    prev_dc: i16,
) -> Result<(usize, i16, Vec<i16>), anyhow::Error> {
    let mut ret = vec!(0; 64);
    let mut offset = 0;

    let (used_bits, dc_sym) = next_sym(&bitstream[offset..], dc_table)?;
    offset += used_bits;
    let num_zeroes = (dc_sym & 0xF0) >> 4;
    assert!(num_zeroes == 0);
    let coeff_len = (dc_sym & 0x0F) as usize;
    assert!(coeff_len <= 11);
    let dc_coeff = next_coeff(&bitstream[offset..], coeff_len);

    offset += coeff_len;
    ret[0] = prev_dc + normalize_coeff(dc_coeff, coeff_len);

    let mut table_pos = 1;
    while table_pos < 64 {
        // eprintln!("  off={} tpos={} bts={} ", offset, table_pos, &bitstream[offset..(offset + 16)]);
        let (used_bits, sym) = next_sym(&bitstream[offset..], ac_table)?;
        offset += used_bits;
        let num_zeroes = ((sym & 0xF0) >> 4) as usize;
        let coeff_len = (sym & 0x0F) as usize;
        assert!(coeff_len <= 10);
        let coeff = next_coeff(&bitstream[offset..], coeff_len);
        // eprintln!("  sym=0x{:02X} num_z={} c_len={} raw={} norm={}", sym, num_zeroes, coeff_len, coeff, normalize_coeff(coeff, coeff_len));

        if num_zeroes == 0 && coeff_len == 0 {
            // eprintln!("  finalize tpos={}, buf={:?}", table_pos, ret);
            return Ok((offset, ret[0], ret));
        }

        if num_zeroes > 0 {
            for i in 0..num_zeroes {
                ret[MCU_ORDER[table_pos + i]] = 0;
            }
            table_pos += num_zeroes;
        }

        offset += coeff_len;
        ret[MCU_ORDER[table_pos]] = normalize_coeff(coeff, coeff_len);
        table_pos += 1;
        assert!(table_pos <= 64);
    }
    Ok((offset, ret[0], ret))
}

fn image_height_mcus(
    height: usize,
    frame_channels: &HashMap<u8, FrameChannel>,
) -> Result<usize, anyhow::Error> {
    let sampling_height = 8 * frame_channels.values().map(|ch| ch.vertical_sampling_factor).max().ok_or(anyhow!("no channels"))?;
    Ok((height + sampling_height as usize - 1) / sampling_height as usize)
}

fn image_width_mcus(
    width: usize,
    frame_channels: &HashMap<u8, FrameChannel>,
) -> Result<usize, anyhow::Error> {
    let sampling_width = 8 * frame_channels.values().map(|ch| ch.horizontal_sampling_factor).max().ok_or(anyhow!("no channels"))?;
    Ok((width + sampling_width as usize - 1) / sampling_width as usize)
}

pub(crate) fn huffman_decode(
    bitstream: &BitVec<u8, Msb0>,
    huffman_tables: &HashMap<HuffmanTableID, HuffmanTable>,
    height: usize,
    width: usize,
    frame_channels: &HashMap<u8, FrameChannel>,
    scan_channels: &HashMap<u8, ScanChannel>,
    channel_order: &Vec<u8>,
) -> Result<Vec<CoefficientMCU>, anyhow::Error> {
    let height_mcus = image_height_mcus(height, frame_channels)?;
    let width_mcus = image_width_mcus(width, frame_channels)?;
    let mut dc_acc = vec!(0i16; 1 + *scan_channels.keys().max().ok_or(anyhow!("no channels"))? as usize);
    let mut ret = Vec::with_capacity(height_mcus * width_mcus);
    let mut offset = 0;

    for _i in 0..(height_mcus * width_mcus) {
        let mut mcu = HashMap::new();
        for channel_id in channel_order {
            let frame_channel = frame_channels.get(&channel_id).ok_or(anyhow!("no frame channel #{}", channel_id))?;
            let scan_channel = scan_channels.get(&channel_id).ok_or(anyhow!("no scan channel #{}", channel_id))?;
            let mut mcu_channel = Vec::new();
            for _j in 0..(frame_channel.horizontal_sampling_factor * frame_channel.vertical_sampling_factor) {
                // eprintln!("hd off={} mcu={} chid={} dc={:?} ac={:?}, rmn={:?}", offset, _i, channel_id, scan_channel.dc_huffman_table_id, scan_channel.ac_huffman_table_id, bitstream.len() - offset);
                let (new_offset, new_dc_acc, block) = block_decode(
                    &bitstream[offset..],
                    huffman_tables.get(&scan_channel.dc_huffman_table_id).ok_or(anyhow!("no ht #{:?}", &scan_channel.dc_huffman_table_id))?,
                    huffman_tables.get(&scan_channel.ac_huffman_table_id).ok_or(anyhow!("no ht #{:?}", &scan_channel.ac_huffman_table_id))?,
                    dc_acc[*channel_id as usize],
                )?;
                offset += new_offset;
                dc_acc[*channel_id as usize] = new_dc_acc;
                mcu_channel.extend_from_slice(&block);
            }
            mcu.insert(*channel_id, mcu_channel);
        }
        ret.push(mcu);
    }
    Ok(ret)
}

pub(crate) fn ycbcr_to_rgb(
    mcus: &Vec<ValueMCU>,
    channels: &HashMap<u8, FrameChannel>,
    channel_order: &Vec<u8>
) -> Vec<RGBMCU> {
    mcus.iter().map(|mcu: &ValueMCU| {
        let mut rgb = HashMap::new();

        let y = mcu.get(&1).unwrap();
        let cb = mcu.get(&2).unwrap();
        let cr = mcu.get(&3).unwrap();

        let max_vsf = channel_order.iter().map(|id| channels.get(id).unwrap().vertical_sampling_factor).max().unwrap() as usize;
        let max_hsf = channel_order.iter().map(|id| channels.get(id).unwrap().horizontal_sampling_factor).max().unwrap() as usize;

        let cb_vsf = channels.get(&2).unwrap().vertical_sampling_factor as usize;
        let cb_hsf = channels.get(&2).unwrap().horizontal_sampling_factor as usize;
        let cr_vsf = channels.get(&3).unwrap().vertical_sampling_factor as usize;
        let cr_hsf = channels.get(&3).unwrap().horizontal_sampling_factor as usize;

        let cb_vsr = max_vsf / cb_vsf;
        let cb_hsr = max_hsf / cb_hsf;
        let cr_vsr = max_vsf / cr_vsf;
        let cr_hsr = max_hsf / cr_hsf;

        let mut r = vec!(0; 64 * max_vsf * max_hsf);
        let mut g = vec!(0; 64 * max_vsf * max_hsf);
        let mut b = vec!(0; 64 * max_vsf * max_hsf);

        let vext = 8 * max_vsf;
        let uext = 8 * max_hsf;

        for v in 0..vext {
            for u in 0..uext {
                let target_range = match (u % uext / 8, v % vext / 8) {
                    (0, 0) => 0..64,
                    (1, 0) => 64..128,
                    (0, 1) => 128..192,
                    (1, 1) => 192..256,
                    _ => unreachable!(),
                };

                let target_y = &y[target_range.clone()];

                let yi = target_y[v % 8 * 8 + u % 8] as f32;
                let cbi = cb[(v / cb_vsr) * 8 * cb_vsf + (u / cb_hsr)] as f32;
                let cri = cr[(v / cr_vsr) * 8 * cr_vsf + (u / cr_hsr)] as f32;

                let target_r = &mut r[target_range.clone()];
                let target_g = &mut g[target_range.clone()];
                let target_b = &mut b[target_range.clone()];

                target_r[v % 8 * 8 + u % 8] = (128.  + yi + 1.402 * cri) as u8;
                target_g[v % 8 * 8 + u % 8] = (128. + yi - 0.34414 * cbi - 0.71414 * cri) as u8;
                target_b[v % 8 * 8 + u % 8] = (128.  + yi + 1.772 * cbi) as u8;
            }
        }

        rgb.insert(1, r);
        rgb.insert(2, g);
        rgb.insert(3, b);
        rgb
    }).collect::<Vec<RGBMCU>>()
}

pub(crate) fn serialize<T>(
    mcus: &Vec<HashMap<u8, Vec<T>>>,
    channels: &HashMap<u8, FrameChannel>,
    channel_order: &Vec<u8>,
    height: usize,
    width: usize,
    uneven: bool,
) -> Vec<T> where T : Copy + Default + std::fmt::Display {
    let img_width_mcus = image_width_mcus(width, channels).unwrap();
    let num_channels = channels.len();
    let mut ret = vec!(T::default(); height * width * num_channels);

    let max_vsf = channel_order.iter().map(|id| channels.get(id).unwrap().vertical_sampling_factor).max().unwrap() as usize;
    let max_hsf = channel_order.iter().map(|id| channels.get(id).unwrap().horizontal_sampling_factor).max().unwrap() as usize;

    for y in 0..height {
        for x in 0..width {
            for (c, channel_id) in channel_order.iter().enumerate() {
                let hsf = channels.get(channel_id).unwrap().horizontal_sampling_factor as usize;
                let vsf = channels.get(channel_id).unwrap().vertical_sampling_factor as usize;
                let sample_width = 8 * if uneven { hsf } else { max_hsf };
                let sample_height = 8 * if uneven { vsf } else { max_vsf };
                let target_mcu = mcus[(y / (8 * max_vsf)) * img_width_mcus + x / (8 * max_hsf)].get(channel_id).unwrap();

                let target_block = match (x % sample_width / 8, y % sample_height / 8) {
                    (0, 0) => &target_mcu[0..64],
                    (1, 0) => &target_mcu[64..128],
                    (0, 1) => &target_mcu[128..192],
                    (1, 1) => &target_mcu[192..256],
                    _ => unreachable!(),
                };

                ret[num_channels * (y * width + x) + c] = target_block[y % 8 * 8 + x % 8];
            }
        }
    }

    return ret;
}

pub fn decode(input: Vec<u8>, show_phases: bool) -> Result<Vec<u8>, anyhow::Error> {
    let jpeg = Jpeg::from_slice(&input)?;

    let mut huffman_decoded = huffman_decode(
        &jpeg.bitstream,
        &jpeg.huffman_tables,
        jpeg.height,
        jpeg.width,
        &jpeg.channels,
        &jpeg.scan.channels,
        &jpeg.channel_order,
    ).unwrap();
    let ycbcr = huffman_decoded.iter_mut().map(|mcu| {
        let dequantized = dequantize(mcu, &jpeg.quant_tables, &jpeg.channels).unwrap();
        inverse_dct(&dequantized, &jpeg.channels, &jpeg.channel_order).unwrap()
    }).collect::<Vec<ValueMCU>>();

    let rgb = ycbcr_to_rgb(&ycbcr, &jpeg.channels, &jpeg.channel_order);

    if show_phases {
        let serialized = serialize(&huffman_decoded, &jpeg.channels, &jpeg.channel_order, jpeg.height, jpeg.width, true)
            .into_iter().map(|x| x as u8).collect::<Vec<u8>>();
        let bmp = write_bmp(&serialized, jpeg.width as u32, jpeg.height as u32);
        std::fs::write("test2-sm.coefficients.bmp", bmp).unwrap();
    }

    if show_phases {
        let serialized = serialize(&ycbcr, &jpeg.channels, &jpeg.channel_order, jpeg.height, jpeg.width, true)
            .into_iter().map(|x| x as u8).collect::<Vec<u8>>();
        let bmp = write_bmp(&serialized, jpeg.width as u32, jpeg.height as u32);
        std::fs::write("test2-sm.ycbcr.bmp", bmp).unwrap();
    }

    {
        let serialized = serialize(&rgb, &jpeg.channels, &jpeg.channel_order, jpeg.height, jpeg.width, false)
            .into_iter().map(|x| x as u8).collect::<Vec<u8>>();
        let bmp = write_bmp(&serialized, jpeg.width as u32, jpeg.height as u32);
        return Ok(bmp);
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_zero_dct() {
        let mcus = HashMap::from([
            (1, vec!(0x00; 64)),
            (2, vec!(0x00; 64)),
            (3, vec!(0x00; 64)),
        ]);

        let channels = HashMap::from([
            (1, FrameChannel { quant_table_id: 0, horizontal_sampling_factor: 1, vertical_sampling_factor: 1 }),
            (2, FrameChannel { quant_table_id: 0, horizontal_sampling_factor: 1, vertical_sampling_factor: 1 }),
            (3, FrameChannel { quant_table_id: 0, horizontal_sampling_factor: 1, vertical_sampling_factor: 1 }),
        ]);

        let channel_order = vec!(1, 2, 3);

        let result = inverse_dct(&mcus, &channels, &channel_order).unwrap();

        assert_eq!(result.get(&1).unwrap(), &vec!(0x00; 64));
        assert_eq!(result.get(&2).unwrap(), &vec!(0x00; 64));
        assert_eq!(result.get(&3).unwrap(), &vec!(0x00; 64));
    }

    #[test]
    fn test_dc_dct() {
        let mcus = HashMap::from([
            (1, vec!(0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00)),
            (2, vec!(-0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00)),
            (3, vec!(0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00)),
        ]);

        assert_eq!(mcus.get(&1).unwrap().len(), 64);
        assert_eq!(mcus.get(&2).unwrap().len(), 64);
        assert_eq!(mcus.get(&3).unwrap().len(), 64);

        let channels = HashMap::from([
            (1, FrameChannel { quant_table_id: 0, horizontal_sampling_factor: 1, vertical_sampling_factor: 1 }),
            (2, FrameChannel { quant_table_id: 0, horizontal_sampling_factor: 1, vertical_sampling_factor: 1 }),
            (3, FrameChannel { quant_table_id: 0, horizontal_sampling_factor: 1, vertical_sampling_factor: 1 }),
        ]);

        let channel_order = vec!(1, 2, 3);

        let result = inverse_dct(&mcus, &channels, &channel_order).unwrap();

        assert_eq!(result.get(&1).unwrap(), &vec!(0x04; 64));
        assert_eq!(result.get(&2).unwrap(), &vec!(-0x03; 64));
        assert_eq!(result.get(&3).unwrap(), &vec!(0x04; 64));
    }

    #[test]
    fn test_2x2_subsampling_dct() {
        let mcus = HashMap::from([
            (1, vec!(
                0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            )),
            (2, vec!(0x00; 64)),
            (3, vec!(0x00; 64)),
        ]);

        assert_eq!(mcus.get(&1).unwrap().len(), 256);
        assert_eq!(mcus.get(&2).unwrap().len(), 64);
        assert_eq!(mcus.get(&3).unwrap().len(), 64);

        let channels = HashMap::from([
            (1, FrameChannel { quant_table_id: 0, horizontal_sampling_factor: 2, vertical_sampling_factor: 2 }),
            (2, FrameChannel { quant_table_id: 0, horizontal_sampling_factor: 1, vertical_sampling_factor: 1 }),
            (3, FrameChannel { quant_table_id: 0, horizontal_sampling_factor: 1, vertical_sampling_factor: 1 }),
        ]);

        let channel_order = vec!(1, 2, 3);

        let result = inverse_dct(&mcus, &channels, &channel_order).unwrap();
        assert_eq!(result.get(&1).unwrap(), &vec!(
            0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04,
            0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08,
            0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04,
            0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08,
        ));
    }

    #[test]
    fn test_zero_ycbcr_to_rgb() {
        let mcus = HashMap::from([
            (1, vec!(0x00; 64)),
            (2, vec!(0x00; 64)),
            (3, vec!(0x00; 64)),
        ]);

        let channels = HashMap::from([
            (1, FrameChannel { quant_table_id: 0, horizontal_sampling_factor: 1, vertical_sampling_factor: 1 }),
            (2, FrameChannel { quant_table_id: 0, horizontal_sampling_factor: 1, vertical_sampling_factor: 1 }),
            (3, FrameChannel { quant_table_id: 0, horizontal_sampling_factor: 1, vertical_sampling_factor: 1 }),
        ]);

        let channel_order = vec!(1, 2, 3);

        let result = ycbcr_to_rgb(&vec!(mcus), &channels, &channel_order);

        assert_eq!(result[0].get(&1).unwrap(), &vec!(0x80; 64));
        assert_eq!(result[0].get(&2).unwrap(), &vec!(0x80; 64));
        assert_eq!(result[0].get(&3).unwrap(), &vec!(0x80; 64));
    }

    #[test]
    fn test_maxlum_ycbcr_to_rgb() {
        let mcus = HashMap::from([
            (1, vec!(0x7F; 64)),
            (2, vec!(0x00; 64)),
            (3, vec!(0x00; 64)),
        ]);

        let channels = HashMap::from([
            (1, FrameChannel { quant_table_id: 0, horizontal_sampling_factor: 1, vertical_sampling_factor: 1 }),
            (2, FrameChannel { quant_table_id: 0, horizontal_sampling_factor: 1, vertical_sampling_factor: 1 }),
            (3, FrameChannel { quant_table_id: 0, horizontal_sampling_factor: 1, vertical_sampling_factor: 1 }),
        ]);

        let channel_order = vec!(1, 2, 3);

        let result = ycbcr_to_rgb(&vec!(mcus), &channels, &channel_order);

        assert_eq!(result[0].get(&1).unwrap(), &vec!(0xFF; 64));
        assert_eq!(result[0].get(&2).unwrap(), &vec!(0xFF; 64));
        assert_eq!(result[0].get(&3).unwrap(), &vec!(0xFF; 64));
    }

    #[test]
    fn test_maxblue_ycbcr_to_rgb() {
        let mcus = HashMap::from([
            (1, vec!(0x00; 64)),
            (2, vec!(0x7F; 64)),
            (3, vec!(0x00; 64)),
        ]);

        let channels = HashMap::from([
            (1, FrameChannel { quant_table_id: 0, horizontal_sampling_factor: 1, vertical_sampling_factor: 1 }),
            (2, FrameChannel { quant_table_id: 0, horizontal_sampling_factor: 1, vertical_sampling_factor: 1 }),
            (3, FrameChannel { quant_table_id: 0, horizontal_sampling_factor: 1, vertical_sampling_factor: 1 }),
        ]);

        let channel_order = vec!(1, 2, 3);

        let result = ycbcr_to_rgb(&vec!(mcus), &channels, &channel_order);

        assert_eq!(result[0].get(&1).unwrap(), &vec!(0x80; 64));
        assert_eq!(result[0].get(&2).unwrap(), &vec!(0x54; 64));
        assert_eq!(result[0].get(&3).unwrap(), &vec!(0xFF; 64));
    }

    #[test]
    fn test_minblue_ycbcr_to_rgb() {
        let mcus = HashMap::from([
            (1, vec!(0x00; 64)),
            (2, vec!(-0x7F; 64)),
            (3, vec!(0x00; 64)),
        ]);

        let channels = HashMap::from([
            (1, FrameChannel { quant_table_id: 0, horizontal_sampling_factor: 1, vertical_sampling_factor: 1 }),
            (2, FrameChannel { quant_table_id: 0, horizontal_sampling_factor: 1, vertical_sampling_factor: 1 }),
            (3, FrameChannel { quant_table_id: 0, horizontal_sampling_factor: 1, vertical_sampling_factor: 1 }),
        ]);

        let channel_order = vec!(1, 2, 3);

        let result = ycbcr_to_rgb(&vec!(mcus), &channels, &channel_order);

        assert_eq!(result[0].get(&1).unwrap(), &vec!(0x80; 64));
        assert_eq!(result[0].get(&2).unwrap(), &vec!(0xAB; 64));
        assert_eq!(result[0].get(&3).unwrap(), &vec!(0x00; 64));
    }

    #[test]
    fn test_maxred_ycbcr_to_rgb() {
        let mcus = HashMap::from([
            (1, vec!(0x00; 64)),
            (2, vec!(0x00; 64)),
            (3, vec!(0x7F; 64)),
        ]);

        let channels = HashMap::from([
            (1, FrameChannel { quant_table_id: 0, horizontal_sampling_factor: 1, vertical_sampling_factor: 1 }),
            (2, FrameChannel { quant_table_id: 0, horizontal_sampling_factor: 1, vertical_sampling_factor: 1 }),
            (3, FrameChannel { quant_table_id: 0, horizontal_sampling_factor: 1, vertical_sampling_factor: 1 }),
        ]);

        let channel_order = vec!(1, 2, 3);

        let result = ycbcr_to_rgb(&vec!(mcus), &channels, &channel_order);

        assert_eq!(result[0].get(&1).unwrap(), &vec!(0xFF; 64));
        assert_eq!(result[0].get(&2).unwrap(), &vec!(0x25; 64));
        assert_eq!(result[0].get(&3).unwrap(), &vec!(0x80; 64));
    }

    #[test]
    fn test_minred_ycbcr_to_rgb() {
        let mcus = HashMap::from([
            (1, vec!(0x00; 64)),
            (2, vec!(0x00; 64)),
            (3, vec!(-0x7F; 64)),
        ]);

        let channels = HashMap::from([
            (1, FrameChannel { quant_table_id: 0, horizontal_sampling_factor: 1, vertical_sampling_factor: 1 }),
            (2, FrameChannel { quant_table_id: 0, horizontal_sampling_factor: 1, vertical_sampling_factor: 1 }),
            (3, FrameChannel { quant_table_id: 0, horizontal_sampling_factor: 1, vertical_sampling_factor: 1 }),
        ]);

        let channel_order = vec!(1, 2, 3);

        let result = ycbcr_to_rgb(&vec!(mcus), &channels, &channel_order);

        assert_eq!(result[0].get(&1).unwrap(), &vec!(0x00; 64));
        assert_eq!(result[0].get(&2).unwrap(), &vec!(0xDA; 64));
        assert_eq!(result[0].get(&3).unwrap(), &vec!(0x80; 64));
    }

    #[test]
    fn test_subsampling_ycbcr_to_rgb_luminance() {
        let mcus = HashMap::from([
            (1, (-0x80..=0x7F).collect()),
            (2, vec!(0x00; 64)),
            (3, vec!(0x00; 64)),
        ]);

        let channels = HashMap::from([
            (1, FrameChannel { quant_table_id: 0, horizontal_sampling_factor: 2, vertical_sampling_factor: 2 }),
            (2, FrameChannel { quant_table_id: 0, horizontal_sampling_factor: 1, vertical_sampling_factor: 1 }),
            (3, FrameChannel { quant_table_id: 0, horizontal_sampling_factor: 1, vertical_sampling_factor: 1 }),
        ]);

        let channel_order = vec!(1, 2, 3);

        let result = ycbcr_to_rgb(&vec!(mcus), &channels, &channel_order);

        assert_eq!(result[0].get(&1).unwrap(), &(0x00..=0xFF).collect::<Vec<u8>>());
        assert_eq!(result[0].get(&2).unwrap(), &(0x00..=0xFF).collect::<Vec<u8>>());
        assert_eq!(result[0].get(&3).unwrap(), &(0x00..=0xFF).collect::<Vec<u8>>());
    }
}
