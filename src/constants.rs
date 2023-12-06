#![allow(non_upper_case_globals)]

use anyhow::anyhow;

pub const MCU_ORDER: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10,
    17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
];

#[repr(u16)]
#[derive(PartialEq, Eq, Debug, Hash, Clone, Copy)]
pub enum Marker {
    SOF0 = 0xFFC0,
    SOF1 = 0xFFC1,
    SOF2 = 0xFFC2,
    SOF3 = 0xFFC3,
    DHT = 0xFFC4,
    SOF5 = 0xFFC5,
    SOF6 = 0xFFC6,
    SOF7 = 0xFFC7,
    JPG = 0xFFC8,
    SOF9 = 0xFFC9,
    SOF10 = 0xFFCA,
    SOF11 = 0xFFCB,
    DAC = 0xFFCC,
    SOF13 = 0xFFCD,
    SOF14 = 0xFFCE,
    SOF15 = 0xFFCF,
    RST0 = 0xFFD0,
    RST1 = 0xFFD1,
    RST2 = 0xFFD2,
    RST3 = 0xFFD3,
    RST4 = 0xFFD4,
    RST5 = 0xFFD5,
    RST6 = 0xFFD6,
    RST7 = 0xFFD7,
    SOI = 0xFFD8,
    EOI = 0xFFD9,
    SOS = 0xFFDA,
    DQT = 0xFFDB,
    DNL = 0xFFDC,
    DRI = 0xFFDD,
    DHP = 0xFFDE,
    EXP = 0xFFDF,
    APP0 = 0xFFE0,
    APP1 = 0xFFE1,
    APP2 = 0xFFE2,
    APP3 = 0xFFE3,
    APP4 = 0xFFE4,
    APP5 = 0xFFE5,
    APP6 = 0xFFE6,
    APP7 = 0xFFE7,
    APP8 = 0xFFE8,
    APP9 = 0xFFE9,
    APP10 = 0xFFEA,
    APP11 = 0xFFEB,
    APP12 = 0xFFEC,
    APP13 = 0xFFED,
    APP14 = 0xFFEE,
    APP15 = 0xFFEF,
    JPG0 = 0xFFF0,
    JPG1 = 0xFFF1,
    JPG2 = 0xFFF2,
    JPG3 = 0xFFF3,
    JPG4 = 0xFFF4,
    JPG5 = 0xFFF5,
    JPG6 = 0xFFF6,
    JPG7 = 0xFFF7,
    JPG8 = 0xFFF8,
    JPG9 = 0xFFF9,
    JPG10 = 0xFFFA,
    JPG11 = 0xFFFB,
    JPG12 = 0xFFFC,
    JPG13 = 0xFFFD,
    COM = 0xFFFE,
    FFFF = 0xFFFF,
}

impl Marker {
    pub fn is_unsupported(self: &Marker) -> bool {
        match self {
            Marker::SOI | Marker::EOI | Marker::DAC | Marker::RST0 | Marker::RST1 |
            Marker::RST2 | Marker::RST3 | Marker::RST4 | Marker::RST5 | Marker::RST6 |
            Marker::RST7 | Marker::SOF1 | Marker::SOF2 | Marker::SOF3 | Marker::SOF5 |
            Marker::SOF6 | Marker::SOF7 | Marker::SOF9 | Marker::SOF10 | Marker::SOF11 |
            Marker::SOF13 | Marker::SOF14 | Marker::SOF15
                => true,
            _
                => false
        }
    }

    pub fn is_ignored(self: &Marker) -> bool {
        match self {
            Marker::APP0 | Marker::APP1 | Marker::APP2 | Marker::APP3 | Marker::APP4 |
            Marker::APP5 | Marker::APP7 | Marker::APP8 | Marker::APP9 | Marker::APP10 |
            Marker::APP11 | Marker::APP12 | Marker::APP13 | Marker::APP14 | Marker::APP15 |
            Marker::JPG0 | Marker::JPG1 | Marker::JPG2 | Marker::JPG3 | Marker::JPG4 |
            Marker::JPG5 | Marker::JPG6 | Marker::JPG7 | Marker::JPG8 | Marker::JPG9 |
            Marker::JPG10 | Marker::JPG11 | Marker::JPG12 | Marker::JPG13 | Marker::DNL |
            Marker::DHP | Marker::EXP | Marker::COM
                => true,
            _
                => false
        }
    }


}

impl TryFrom<u16> for Marker {
    type Error = anyhow::Error;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        match value {
            0xFFC0 => Ok(Marker::SOF0),
            0xFFC1 => Ok(Marker::SOF1),
            0xFFC2 => Ok(Marker::SOF2),
            0xFFC3 => Ok(Marker::SOF3),
            0xFFC4 => Ok(Marker::DHT),
            0xFFC5 => Ok(Marker::SOF5),
            0xFFC6 => Ok(Marker::SOF6),
            0xFFC7 => Ok(Marker::SOF7),
            0xFFC8 => Ok(Marker::JPG),
            0xFFC9 => Ok(Marker::SOF9),
            0xFFCA => Ok(Marker::SOF10),
            0xFFCB => Ok(Marker::SOF11),
            0xFFCC => Ok(Marker::DAC),
            0xFFCD => Ok(Marker::SOF13),
            0xFFCE => Ok(Marker::SOF14),
            0xFFCF => Ok(Marker::SOF15),
            0xFFD0 => Ok(Marker::RST0),
            0xFFD1 => Ok(Marker::RST1),
            0xFFD2 => Ok(Marker::RST2),
            0xFFD3 => Ok(Marker::RST3),
            0xFFD4 => Ok(Marker::RST4),
            0xFFD5 => Ok(Marker::RST5),
            0xFFD6 => Ok(Marker::RST6),
            0xFFD7 => Ok(Marker::RST7),
            0xFFD8 => Ok(Marker::SOI),
            0xFFD9 => Ok(Marker::EOI),
            0xFFDA => Ok(Marker::SOS),
            0xFFDB => Ok(Marker::DQT),
            0xFFDC => Ok(Marker::DNL),
            0xFFDD => Ok(Marker::DRI),
            0xFFDE => Ok(Marker::DHP),
            0xFFDF => Ok(Marker::EXP),
            0xFFE0 => Ok(Marker::APP0),
            0xFFE1 => Ok(Marker::APP1),
            0xFFE2 => Ok(Marker::APP2),
            0xFFE3 => Ok(Marker::APP3),
            0xFFE4 => Ok(Marker::APP4),
            0xFFE5 => Ok(Marker::APP5),
            0xFFE6 => Ok(Marker::APP6),
            0xFFE7 => Ok(Marker::APP7),
            0xFFE8 => Ok(Marker::APP8),
            0xFFE9 => Ok(Marker::APP9),
            0xFFEA => Ok(Marker::APP10),
            0xFFEB => Ok(Marker::APP11),
            0xFFEC => Ok(Marker::APP12),
            0xFFED => Ok(Marker::APP13),
            0xFFEE => Ok(Marker::APP14),
            0xFFEF => Ok(Marker::APP15),
            0xFFF0 => Ok(Marker::JPG0),
            0xFFF1 => Ok(Marker::JPG1),
            0xFFF2 => Ok(Marker::JPG2),
            0xFFF3 => Ok(Marker::JPG3),
            0xFFF4 => Ok(Marker::JPG4),
            0xFFF5 => Ok(Marker::JPG5),
            0xFFF6 => Ok(Marker::JPG6),
            0xFFF7 => Ok(Marker::JPG7),
            0xFFF8 => Ok(Marker::JPG8),
            0xFFF9 => Ok(Marker::JPG9),
            0xFFFA => Ok(Marker::JPG10),
            0xFFFB => Ok(Marker::JPG11),
            0xFFFC => Ok(Marker::JPG12),
            0xFFFD => Ok(Marker::JPG13),
            0xFFFE => Ok(Marker::COM),
            0xFFFF => Ok(Marker::FFFF),
            _ => Err(anyhow!(format!("Unknown value for Marker: {:X}", value))),
        }
    }
}

const m0: f32 = 1.847759;
pub const m1: f32 = 1.4142135;
pub const m3: f32 = m1;
pub const m5: f32 = 0.76536685;
pub const m2: f32 = m0 - m5;
pub const m4: f32 = m0 + m5;

pub const s0: f32 = 0.35355338;
pub const s1: f32 = 0.49039263;
pub const s2: f32 = 0.46193975;
pub const s3: f32 = 0.4157348;
pub const s4: f32 = 0.35355338;
pub const s5: f32 = 0.2777851;
pub const s6: f32 = 0.19134171;
pub const s7: f32 = 0.09754512;
