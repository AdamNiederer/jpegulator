use std::collections::{HashMap, HashSet};
use anyhow::anyhow;
use bitvec::prelude::*;
use smallvec::SmallVec;
use crate::constants::{Marker, MCU_ORDER};

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum HuffmanTableID {
    AC(usize),
    DC(usize),
}

#[derive(Debug, Eq, PartialEq)]
pub struct FrameChannel {
    pub quant_table_id: usize,
    pub horizontal_sampling_factor: u8,
    pub vertical_sampling_factor: u8,
}

#[derive(Debug)]
pub struct Frame {
    pub height: usize,
    pub width: usize,
    pub depth: usize,
    pub channel_order: Vec<u8>,
    pub channels: HashMap<u8, FrameChannel>,
}

#[derive(Debug, Eq, PartialEq)]
pub struct HuffmanTable {
    pub offsets: SmallVec<u8, 16>,
    pub lengths: SmallVec<u8, 16>,
    pub symbols: SmallVec<u8, 162>,
    pub codes: SmallVec<u16, 162>,
    pub min_code_len: u8,
    pub max_code_len: u8,
}

impl HuffmanTable {
    fn new() -> HuffmanTable {
        return HuffmanTable {
            offsets: SmallVec::with_capacity(16),
            lengths: SmallVec::with_capacity(16),
            symbols: SmallVec::with_capacity(162),
            codes: SmallVec::with_capacity(162),
            min_code_len: 0,
            max_code_len: 0,
        };
    }
}

#[derive(Debug)]
pub struct Jpeg {
    pub quant_tables: HashMap<usize, Vec<u16>>,
    pub channels: HashMap<u8, FrameChannel>,
    pub channel_order: Vec<u8>,
    pub huffman_tables: HashMap<HuffmanTableID, HuffmanTable>,
    pub restart_interval: usize,
    pub height: usize,
    pub width: usize,
    pub scan: Scan,
    pub bitstream: BitVec<u8, Msb0>,
}

#[derive(Debug)]
pub struct Scan {
    pub first_dct_coeff: u8,
    pub last_dct_coeff: u8,
    pub successive_approx_high: u8,
    pub successive_approx_low: u8,
    pub channels: HashMap<u8, ScanChannel>,
}

#[derive(Debug)]
pub struct ScanChannel {
    pub ac_huffman_table_id: HuffmanTableID,
    pub dc_huffman_table_id: HuffmanTableID,
}

pub type CoefficientMCU = HashMap<u8, Vec<i16>>;
pub type ValueMCU = HashMap<u8, Vec<i8>>;
pub type RGBMCU = HashMap<u8, Vec<u8>>;

fn parse_quantization_table(buf: &[u8]) -> HashMap<usize, Vec<u16>> {
    let mut offset = 1;
    let mut ret = HashMap::new();
    while offset < buf.len() {
        let is_16bit = (buf[offset - 1] & 0xF0) != 0;
        let id = (buf[offset - 1] & 0x0F) as usize;
        assert!(id < 3);
        let mut table = vec!(0; 64);
        if is_16bit {
            for i in 0..64 {
                table[MCU_ORDER[i]] = get_u16(buf, offset + 2 * i)
            }
        } else {
            for i in 0..64 {
                table[MCU_ORDER[i]] = buf[offset + i] as u16
            }
        }
        assert!(!ret.contains_key(&id));
        ret.insert(id, table);
        offset += 1 + if is_16bit { 128 } else { 64 }
    }
    return ret;
}

fn parse_restart_interval(buf: &[u8]) -> usize {
    return get_u16(buf, 0) as usize;
}

fn parse_start_of_frame_0(buf: &[u8]) -> Frame {
    let mut ret = Frame {
        depth: buf[0] as usize,
        height: get_u16(buf, 1) as usize,
        width: get_u16(buf, 3) as usize,
        channels: HashMap::new(),
        channel_order: Vec::new(),
    };
    assert!(ret.depth == 8);
    let channels = buf[5] as usize;
    assert!(channels > 0);
    for i in 0..channels {
        let id = buf[6 + 3 * i];
        ret.channel_order.push(id);
        assert!(id > 0 && id <= 3); // only support YCbCr
        assert!(!ret.channels.contains_key(&id));
        ret.channels.insert(id, FrameChannel {
            horizontal_sampling_factor: (buf[6 + 3 * i + 1] & 0xF0) >> 4,
            vertical_sampling_factor: buf[6 + 3 * i + 1] & 0x0F,
            quant_table_id: buf[6 + 3 * i + 2] as usize,
        });
    }
    return ret;
}

fn huffman_codes(lengths: &[u8]) -> SmallVec<u16, 162> {
    let mut code = 0;
    let mut ret = SmallVec::<u16, 162>::with_capacity(162);
    for i in 0..lengths.len() {
        for _ in 0..lengths[i] {
            ret.push(code);
            code += 1;
        }
        code = code << 1;
    }
    assert!(lengths.iter().fold(0, |acc, it| acc + it) <= 162);
    assert!(ret.len() <= 162);
    return ret;
}

fn parse_huffman_table(buf: &[u8]) -> HashMap<HuffmanTableID, HuffmanTable> {
    let mut offset = 0;
    let mut ret = HashMap::new();
    while offset < buf.len() {
        let is_ac = (buf[offset] & 0xF0) != 0;
        let id = (buf[offset] & 0x0F) as usize;
        assert!(id <= 3);
        let mut sym_count = 0;
        let mut table = HuffmanTable::new();
        let count_segment = &buf[(offset + 1)..(offset + 17)];
        for sym_len_count in count_segment {
            table.lengths.push(*sym_len_count);

            if *sym_len_count > 0 || sym_count > 0 {
                table.offsets.push(sym_count);
            }

            sym_count += sym_len_count;
        }

        assert!(sym_count <= 162);
        assert!(sym_count > 0);

        table.min_code_len = 1 + table.lengths.iter().position(|len| *len > 0).unwrap() as u8;
        table.max_code_len = 1 + table.lengths.iter().rposition(|len| *len > 0).unwrap() as u8;
        table.offsets.truncate((1 + table.max_code_len - table.min_code_len) as usize);

        table.symbols.extend_from_slice(&buf[(offset + 17)..(offset + 17 + sym_count as usize)]);

        // All symbols should be unique
        let mut uniq_check = HashSet::new();
        assert!(table.symbols.iter().all(|item| uniq_check.insert(item)));

        for sym in &table.symbols {
            assert!(sym & 0x0F <= if is_ac { 10 } else { 11 });
        }

        table.codes = huffman_codes(&table.lengths);

        ret.insert(if is_ac { HuffmanTableID::AC(id) } else { HuffmanTableID::DC(id) }, table);
        offset += 1 + 16 + sym_count as usize;
    }
    return ret;
}

fn parse_start_of_scan(buf: &[u8]) -> Scan {
    let _num_channels = buf[0];
    let first_dct_coeff = buf[buf.len() - 3];
    assert!(first_dct_coeff == 0);
    let last_dct_coeff = buf[buf.len() - 2];
    assert!(last_dct_coeff == 63);
    let successive_approx_high = (buf[buf.len() - 1] & 0xF0) >> 4;
    let successive_approx_low = buf[buf.len() - 1] & 0x0F;
    assert!(successive_approx_high == 0);
    assert!(successive_approx_low == 0);
    let mut offset = 1;
    let mut channels = HashMap::new();
    while offset < buf.len() - 3 {
        let id = buf[offset];
        channels.insert(id, ScanChannel {
            ac_huffman_table_id: HuffmanTableID::AC((buf[offset + 1] & 0x0F) as usize),
            dc_huffman_table_id: HuffmanTableID::DC(((buf[offset + 1] & 0xF0) >> 4) as usize),
        });
        offset += 2
    }
    Scan {
        first_dct_coeff,
        last_dct_coeff,
        successive_approx_high,
        successive_approx_low,
        channels,
    }
}

fn parse_bitstream(buf: &[u8]) -> Result<BitVec<u8, Msb0>, anyhow::Error> {
    let mut bits = Vec::<u8>::with_capacity(buf.len());
    let mut offset = 0;
    while offset < buf.len() {
        if offset == 0xFF && offset == buf.len() - 1 {
            return Err(anyhow!("0xFF at end of scan"));
        } else if buf[offset] == 0xFF && buf[offset + 1] == 0xFF {
            offset += 1;
        } else if buf[offset] == 0xFF && buf[offset + 1] == 0x00 {
            bits.push(0xFF);
            offset += 2;
        } else if buf[offset] == 0xFF {
            let marker = get_u16(buf, offset) as u16;
            match Marker::try_from(marker)? {
                parsed @ (Marker::RST0 | Marker::RST1 | Marker::RST2 | Marker::RST3 | Marker::RST4 | Marker::RST5 | Marker::RST6 | Marker::RST7 | Marker::EOI) => {
                    eprintln!("Ignoring {:?} marker in bitstream @ {}", parsed, offset);
                    offset += 2;
                },
                parsed => {
                    return Err(anyhow!("Unknown marker in bitstream: 0x{:X} ({:?})", marker, parsed));
                }
            }
            offset += 2;
        } else {
            bits.push(buf[offset]);
            offset += 1;
        }
    }

    return Ok(BitVec::<u8, Msb0>::from_vec(bits));
}

fn get_u16(buf: &[u8], offset: usize) -> u16 {
    return ((buf[offset] as u16) << 8) | buf[offset + 1] as u16;
}

fn marker_body(buf: &[u8], offset: usize, length: usize) -> &[u8] {
    return &buf[(offset + 4)..(offset + length + 2)]
}

impl Jpeg {
    pub fn from_slice(file: &[u8]) -> Result<Self, anyhow::Error> {
        Marker::try_from(get_u16(file, 0)).or(Err(anyhow!("No SOI at image start")))?;
        let mut offset = 2;
        let mut ret = Self {
            quant_tables: HashMap::new(),
            huffman_tables: HashMap::new(),
            channels: HashMap::new(),
            channel_order: Vec::new(),
            restart_interval: 0,
            height: 0,
            width: 0,
            scan: Scan { successive_approx_high: 0, successive_approx_low: 0, last_dct_coeff: 0, first_dct_coeff: 0, channels: HashMap::new(), },
            bitstream: BitVec::new(),
        };

        while offset < file.len() {
            let marker = Marker::try_from(get_u16(file, offset))?;

            if marker == Marker::FFFF {
                offset += 1;
                continue;
            } else if marker.is_unsupported() {
                return Err(anyhow!("Unsupported marker: {:?}", marker));
            }

            let length = get_u16(file, offset + 2) as usize;
            match Marker::try_from(marker) {
                Ok(Marker::DRI) => {
                    ret.restart_interval = parse_restart_interval(&marker_body(file, offset, length))
                }
                Ok(Marker::DHT) => {
                    ret.huffman_tables.extend(parse_huffman_table(&marker_body(file, offset, length)))
                }
                Ok(Marker::DQT) => {
                    ret.quant_tables.extend(parse_quantization_table(&marker_body(file, offset, length)))
                }
                Ok(Marker::SOF0) => {
                    let frame = parse_start_of_frame_0(&marker_body(file, offset, length));
                    ret.height = frame.height;
                    ret.width = frame.width;
                    ret.channels.extend(frame.channels);
                    ret.channel_order.extend(frame.channel_order);
                }
                Ok(Marker::SOS) => {
                    ret.scan = parse_start_of_scan(&marker_body(file, offset, length));
                    offset += length as usize + 2;
                    break;
                }
                Ok(_) => { if !marker.is_ignored() { return Err(anyhow!("Unhandled marker {:?}", marker)) } }
                Err(_) => { return Err(anyhow!("Invalid marker @ offset 0x{:X}", offset)) }
            }
            offset += length as usize + 2;

        }

        ret.bitstream = parse_bitstream(&file[offset..])?;
        return Ok(ret);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    

    #[test]
    fn test_parse_dc_huffman_table() {
        let basic_ht = vec!(
            0x00, 0x00, 0x00, 0x05, 0x05, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x06, 0x07, 0x08, 0x09,
            0x01, 0x03, 0x04, 0x05, 0x0a, 0x00
        );

        assert_eq!(basic_ht.len(), 28);

        let actual = parse_huffman_table(&basic_ht);

        let Some(key) = actual.keys().nth(0) else { panic!() };
        assert!(actual.len() == 1);
        assert_eq!(*key, HuffmanTableID::DC(0));

        let entry = actual.get(key).unwrap();
        assert_eq!(entry.lengths.len(), 16);
        assert_eq!(entry.lengths, SmallVec::<_, 16>::from(&basic_ht[1..17]));
        assert_eq!(entry.offsets, SmallVec::<_, 162>::from(vec!(0x00, 0x05, 0x0a)));
        assert_eq!(entry.min_code_len, 3);
        assert_eq!(entry.max_code_len, 5);
        assert_eq!(entry.symbols.len(), 11);
        assert_eq!(entry.symbols, SmallVec::<_, 162>::from(vec!(
            0x02, 0x06, 0x07, 0x08, 0x09, 0x01, 0x03, 0x04, 0x05, 0x0a, 0x00
        )));
    }

    #[test]
    fn test_parse_ac_huffman_table() {
        let ac_ht = vec!(
            0x10, 0x00, 0x01, 0x03, 0x03, 0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x03,
            0x0a, 0x03, 0x04, 0x01, 0x15, 0x02, 0x01, 0x03, 0x04, 0x05, 0x06, 0x11,
            0x00, 0x07, 0x12, 0x08, 0x13, 0x21, 0x22, 0x31, 0x09, 0x14, 0x32, 0x41,
            0x15, 0x42, 0x51, 0x16, 0x23, 0x52, 0x61, 0x71, 0x62, 0x81, 0x91, 0x17,
            0x24, 0x33, 0x72, 0x82, 0xa1, 0xb1, 0xc1, 0xd1, 0xf0, 0x92, 0xa2, 0xe1,
            0x0a, 0x18, 0x25, 0x34, 0x43, 0xb2, 0xf1, 0x19, 0x26, 0xc2, 0x53, 0xe2,
            0x35, 0x44, 0x63, 0x73, 0xd2, 0xf2, 0x27, 0x36, 0x45, 0x54, 0x83, 0x37,
            0x56, 0x64
        );

        assert_eq!(ac_ht.len(), 86);
        let actual = parse_huffman_table(&ac_ht);

        let Some(key) = actual.keys().nth(0) else { panic!() };
        assert!(actual.len() == 1);
        assert_eq!(*key, HuffmanTableID::AC(0));

        let entry = actual.get(key).unwrap();
        assert_eq!(entry.lengths.len(), 16);
        assert_eq!(entry.lengths, SmallVec::<_, 16>::from(&ac_ht[1..17]));
        assert_eq!(entry.offsets, SmallVec::<_, 162>::from(vec!(0x00, 0x01, 0x04, 0x07, 0x0a, 0x0d, 0x0f, 0x13, 0x16, 0x1b, 0x1e, 0x28, 0x2b, 0x2f, 0x30)));
        assert_eq!(entry.min_code_len, 2);
        assert_eq!(entry.max_code_len, 16);
        assert_eq!(entry.symbols.len(), 0x45);
        assert_eq!(entry.symbols, SmallVec::<_, 162>::from(vec!(
            0x02, 0x01, 0x03, 0x04, 0x05, 0x06, 0x11, 0x00, 0x07, 0x12, 0x08,
            0x13, 0x21, 0x22, 0x31, 0x09, 0x14, 0x32, 0x41, 0x15, 0x42, 0x51,
            0x16, 0x23, 0x52, 0x61, 0x71, 0x62, 0x81, 0x91, 0x17, 0x24, 0x33,
            0x72, 0x82, 0xa1, 0xb1, 0xc1, 0xd1, 0xf0, 0x92, 0xa2, 0xe1, 0x0a,
            0x18, 0x25, 0x34, 0x43, 0xb2, 0xf1, 0x19, 0x26, 0xc2, 0x53, 0xe2,
            0x35, 0x44, 0x63, 0x73, 0xd2, 0xf2, 0x27, 0x36, 0x45, 0x54, 0x83,
            0x37, 0x56, 0x64
        )));
    }

    #[test]
    fn test_parse_dc_chroma_huffman_table() {
        let ac_ht = vec!(
            0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04,
            0x05
        );

        assert_eq!(ac_ht.len(), 23);
        let actual = parse_huffman_table(&ac_ht);

        let Some(key) = actual.keys().nth(0) else { panic!() };
        assert!(actual.len() == 1);
        assert_eq!(*key, HuffmanTableID::DC(1));

        let entry = actual.get(key).unwrap();
        assert_eq!(entry.lengths.len(), 16);
        assert_eq!(entry.lengths, SmallVec::<_, 16>::from(&ac_ht[1..17]));
        assert_eq!(entry.offsets, SmallVec::<_, 162>::from(vec!(0x00, 0x01, 0x02, 0x03, 0x04, 0x05)));
        assert_eq!(entry.min_code_len, 1);
        assert_eq!(entry.max_code_len, 6);
        assert_eq!(entry.symbols.len(), 6);
        assert_eq!(entry.symbols, SmallVec::<_, 162>::from(vec!(0x00, 0x01, 0x02, 0x03, 0x04, 0x05)));
    }

    #[test]
    fn test_parse_ac_chroma_huffman_table() {
        let ac_ht = vec!(
            0x11, 0x01, 0x01, 0x01, 0x00, 0x02, 0x02, 0x02, 0x03, 0x01, 0x01,
            0x00, 0x02, 0x01, 0x04, 0x03, 0x01, 0x00, 0x01, 0x11, 0x21, 0x31,
            0x02, 0x41, 0x12, 0x51, 0x22, 0x32, 0x61, 0x71, 0x42, 0x03, 0x81,
            0x13, 0x23, 0x52, 0xa1, 0xb1, 0x33, 0x43, 0x91, 0xf0,
        );

        assert_eq!(ac_ht.len(), 42);
        let actual = parse_huffman_table(&ac_ht);

        let Some(key) = actual.keys().nth(0) else { panic!() };
        assert!(actual.len() == 1);
        assert_eq!(*key, HuffmanTableID::AC(1));

        let entry = actual.get(key).unwrap();
        assert_eq!(entry.lengths.len(), 16);
        assert_eq!(entry.lengths, SmallVec::<_, 16>::from(&ac_ht[1..17]));
        assert_eq!(entry.offsets, SmallVec::<_, 162>::from(vec!(
            0x00, 0x01, 0x02, 0x03, 0x03, 0x05, 0x07, 0x09, 0x0c, 0x0d, 0x0e,
            0x0e, 0x10, 0x11, 0x15, 0x18
        )));
        assert_eq!(entry.min_code_len, 1);
        assert_eq!(entry.max_code_len, 16);
        assert_eq!(entry.symbols.len(), 25);
        assert_eq!(entry.symbols, SmallVec::<_, 162>::from(vec!(
            0x00, 0x01, 0x11, 0x21, 0x31, 0x02, 0x41, 0x12, 0x51, 0x22, 0x32,
            0x61, 0x71, 0x42, 0x03, 0x81, 0x13, 0x23, 0x52, 0xa1, 0xb1, 0x33,
            0x43, 0x91, 0xf0,
        )));
    }

    #[test]
    fn test_huffman_codes() {
        let simple_count = vec!(0, 2, 1, 3);
        let actual = huffman_codes(&simple_count);
        assert_eq!(actual, SmallVec::<_, 162>::from(vec!(0b00, 0b01, 0b100, 0b1010, 0b1011, 0b1100)));
    }

    #[test]
    fn test_sof() {
        let sof = vec!(
            0x08, 0x07, 0x80, 0x05, 0xa0, 0x03, 0x01, 0x22, 0x00, 0x02, 0x11,
            0x01, 0x03, 0x11, 0x01,
        );
        assert_eq!(sof.len(), 15);
        let actual = parse_start_of_frame_0(&sof);
        assert_eq!(actual.width, 1440);
        assert_eq!(actual.height, 1920);
        assert_eq!(actual.channels.len(), 3);
        assert_eq!(actual.channels.get(&1).unwrap().horizontal_sampling_factor, 2);
        assert_eq!(actual.channels.get(&1).unwrap().vertical_sampling_factor, 2);
        assert_eq!(actual.channels.get(&1).unwrap().quant_table_id, 0x00);
        assert_eq!(actual.channels.get(&2).unwrap().horizontal_sampling_factor, 1);
        assert_eq!(actual.channels.get(&2).unwrap().vertical_sampling_factor, 1);
        assert_eq!(actual.channels.get(&2).unwrap().quant_table_id, 0x01);
        assert_eq!(actual.channels.get(&3).unwrap().horizontal_sampling_factor, 1);
        assert_eq!(actual.channels.get(&3).unwrap().vertical_sampling_factor, 1);
        assert_eq!(actual.channels.get(&3).unwrap().quant_table_id, 0x01);
    }

    #[test]
    fn test_sos() {
        let sos = vec!(
            0x03, 0x01, 0x00, 0x02, 0x11, 0x03, 0x11, 0x00, 0x3f, 0x00
        );
        assert_eq!(sos.len(), 10);
        let actual = parse_start_of_scan(&sos);

        assert_eq!(actual.channels.len(), 3);
        assert_eq!(actual.channels.get(&1).unwrap().ac_huffman_table_id, HuffmanTableID::AC(0));
        assert_eq!(actual.channels.get(&1).unwrap().dc_huffman_table_id, HuffmanTableID::DC(0));
        assert_eq!(actual.channels.get(&2).unwrap().ac_huffman_table_id, HuffmanTableID::AC(1));
        assert_eq!(actual.channels.get(&2).unwrap().dc_huffman_table_id, HuffmanTableID::DC(1));
        assert_eq!(actual.channels.get(&3).unwrap().ac_huffman_table_id, HuffmanTableID::AC(1));
        assert_eq!(actual.channels.get(&3).unwrap().dc_huffman_table_id, HuffmanTableID::DC(1));
        assert_eq!(actual.first_dct_coeff, 0);
        assert_eq!(actual.last_dct_coeff, 63);
        assert_eq!(actual.successive_approx_high, 0);
        assert_eq!(actual.successive_approx_low, 0);
    }

    #[test]
    fn test_bitstream() {
        let bits = vec!(
            0x01, 0x02, 0x03, 0xFF, 0xFF, 0x00, 0xFF, 0x00, 0x04, 0x05
        );
        let Ok(actual) = parse_bitstream(&bits) else { panic!(); };

        assert_eq!(actual, BitVec::<u8, Msb0>::from_slice(&vec!(
           0x01, 0x02, 0x03, 0xFF, 0xFF, 0x04, 0x05
        )));
    }
}
