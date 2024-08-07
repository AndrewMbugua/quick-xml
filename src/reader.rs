//! A module to handle `Reader`

#[cfg(feature = "encoding")]
use std::borrow::Cow;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;
use std::str::from_utf8;

#[cfg(feature = "encoding")]
use encoding_rs::{Encoding, UTF_16BE, UTF_16LE};

use crate::errors::{Error, Result};
use crate::events::{attributes::Attribute, BytesDecl, BytesEnd, BytesStart, BytesText, Event};

use memchr;

#[derive(Clone)]
enum TagState {
    /// Nothing has been parsed yet.
    Beginning,

    /// An empty (self-closing) tag has been encountered and the current
    /// configuration asks us to expand them (send two events: an opening and a
    /// closing one, instead of a single one: an `Empty` event)
    Empty,

    /// Errored
    Exit,

    /// A closing tag ">" has been encountered.
    /// `offset` is the offset of the closing tag in the linked buffer.
    ClosingTagEncountered { offset: usize },

    /// An Opening tag "<" has been encountered.
    /// `offset` is the offset of the opening tag in the linked buffer.
    OpeningTagEncountered {
        offset: usize,
        reading_state: State,
    },

    /// We're currently inside a comment.
    /// `offset` is the offset of the opening tag in the linked buffer.
    InsideComment { offset: usize },

    /// We're currently inside a CDATA section.
    /// `offset` is the offset of the opening tag in the linked buffer.
    InsideCdata { offset: usize },

    /// We're currently inside a Document Type Definition.
    /// `offset` is the offset of the opening tag in the linked buffer.
    InsideDocType { offset: usize },
}

/// A low level encoding-agnostic XML event reader.
///
/// Consumes a `BufRead` and streams XML `Event`s.
///
/// # Examples
///
/// ```
/// use quick_xml::Reader;
/// use quick_xml::events::Event;
///
/// let xml = r#"<tag1 att1 = "test">
///                 <tag2><!--Test comment-->Test</tag2>
///                 <tag2>Test 2</tag2>
///             </tag1>"#;
/// let mut reader = Reader::from_str(xml);
/// reader.trim_text(true);
/// let mut count = 0;
/// let mut txt = Vec::new();
/// let mut buf = Vec::new();
/// loop {
///     match reader.read_event(&mut buf) {
///         Ok(Event::Start(ref e)) => {
///             match e.name() {
///                 b"tag1" => println!("attributes values: {:?}",
///                                     e.attributes().map(|a| a.unwrap().value)
///                                     .collect::<Vec<_>>()),
///                 b"tag2" => count += 1,
///                 _ => (),
///             }
///         },
///         Ok(Event::Text(e)) => txt.push(e.unescape_and_decode(&reader).unwrap()),
///         Err(e) => panic!("Error at position {}: {:?}", reader.buffer_position(), e),
///         Ok(Event::Eof) => break,
///         _ => (),
///     }
///     buf.clear();
/// }
/// ```
#[derive(Clone)]
pub struct Reader<B: BufRead> {
    /// reader
    reader: B,
    /// current buffer position, useful for debuging errors
    buf_position: usize,
    /// Last "state" milestone encountered.
    last_tag_state: TagState,
    /// expand empty element into an opening and closing element
    expand_empty_elements: bool,
    /// trims leading whitespace in Text events, skip the element if text is empty
    trim_text_start: bool,
    /// trims trailing whitespace in Text events.
    trim_text_end: bool,
    /// trims trailing whitespaces from markup names in closing tags `</a >`
    trim_markup_names_in_closing_tags: bool,
    /// check if End nodes match last Start node
    check_end_names: bool,
    /// check if comments contains `--` (false per default)
    check_comments: bool,
    /// all currently Started elements which didn't have a matching
    /// End element yet
    opened_buffer: Vec<u8>,
    /// opened name start indexes
    opened_starts: Vec<usize>,
    /// a buffer to manage namespaces
    ns_buffer: NamespaceBufferIndex,
    #[cfg(feature = "encoding")]
    /// the encoding specified in the xml, defaults to utf8
    encoding: &'static Encoding,
    #[cfg(feature = "encoding")]
    /// check if quick-rs could find out the encoding
    is_encoding_set: bool,
}

impl<B: BufRead> Reader<B> {
    /// Creates a `Reader` that reads from a reader implementing `BufRead`.
    pub fn from_reader(reader: B) -> Reader<B> {
        Reader {
            reader,
            opened_buffer: Vec::new(),
            opened_starts: Vec::new(),
            last_tag_state: TagState::Beginning,
            expand_empty_elements: false,
            trim_text_start: false,
            trim_text_end: false,
            trim_markup_names_in_closing_tags: true,
            check_end_names: true,
            buf_position: 0,
            check_comments: false,
            ns_buffer: NamespaceBufferIndex::default(),
            #[cfg(feature = "encoding")]
            encoding: ::encoding_rs::UTF_8,
            #[cfg(feature = "encoding")]
            is_encoding_set: false,
        }
    }

    /// Changes whether empty elements should be split into an `Open` and a `Close` event.
    ///
    /// When set to `true`, all [`Empty`] events produced by a self-closing tag like `<tag/>` are
    /// expanded into a [`Start`] event followed by a [`End`] event. When set to `false` (the
    /// default), those tags are represented by an [`Empty`] event instead.
    ///
    /// (`false` by default)
    ///
    /// [`Empty`]: events/enum.Event.html#variant.Empty
    /// [`Start`]: events/enum.Event.html#variant.Start
    /// [`End`]: events/enum.Event.html#variant.End
    pub fn expand_empty_elements(&mut self, val: bool) -> &mut Reader<B> {
        self.expand_empty_elements = val;
        self
    }

    /// Changes whether whitespace before and after character data should be removed.
    ///
    /// When set to `true`, all [`Text`] events are trimmed. If they are empty, no event will be
    /// pushed.
    ///
    /// (`false` by default)
    ///
    /// [`Text`]: events/enum.Event.html#variant.Text
    pub fn trim_text(&mut self, val: bool) -> &mut Reader<B> {
        self.trim_text_start = val;
        self.trim_text_end = val;
        self
    }

    /// Changes whether whitespace after character data should be removed.
    ///
    /// When set to `true`, trailing whitespace is trimmed in [`Text`] events.
    ///
    /// (`false` by default)
    ///
    /// [`Text`]: events/enum.Event.html#variant.Text
    pub fn trim_text_end(&mut self, val: bool) -> &mut Reader<B> {
        self.trim_text_end = val;
        self
    }

    /// Changes whether trailing whitespaces after the markup name are trimmed in closing tags
    /// `</a >`.
    ///
    /// If true the emitted [`End`] event is stripped of trailing whitespace after the markup name.
    ///
    /// Note that if set to `false` and `check_end_names` is true the comparison of markup names is
    /// going to fail erronously if a closing tag contains trailing whitespaces.
    ///
    /// (`true` by default)
    ///
    /// [`End`]: events/enum.Event.html#variant.End
    pub fn trim_markup_names_in_closing_tags(&mut self, val: bool) -> &mut Reader<B> {
        self.trim_markup_names_in_closing_tags = val;
        self
    }

    /// Changes whether mismatched closing tag names should be detected.
    ///
    /// When set to `false`, it won't check if a closing tag matches the corresponding opening tag.
    /// For example, `<mytag></different_tag>` will be permitted.
    ///
    /// If the XML is known to be sane (already processed, etc.) this saves extra time.
    ///
    /// Note that the emitted [`End`] event will not be modified if this is disabled, ie. it will
    /// contain the data of the mismatched end tag.
    ///
    /// (`true` by default)
    ///
    /// [`End`]: events/enum.Event.html#variant.End
    pub fn check_end_names(&mut self, val: bool) -> &mut Reader<B> {
        self.check_end_names = val;
        self
    }

    /// Changes whether comments should be validated.
    ///
    /// When set to `true`, every [`Comment`] event will be checked for not containing `--`, which
    /// is not allowed in XML comments. Most of the time we don't want comments at all so we don't
    /// really care about comment correctness, thus the default value is `false` to improve
    /// performance.
    ///
    /// (`false` by default)
    ///
    /// [`Comment`]: events/enum.Event.html#variant.Comment
    pub fn check_comments(&mut self, val: bool) -> &mut Reader<B> {
        self.check_comments = val;
        self
    }

    /// Gets the currently considered position in the input data.
    ///
    /// Useful when debugging errors.
    pub fn buffer_position(&self) -> usize {
        // when internal state is Opened, we have actually read until '<',
        // which we don't want to show
        self.buf_position
    }

    /// Private function which will read until encountering the first '<'
    /// Returns a `Text` event if text is encountered while doing so.
    fn read_until_opening_tag<'a, 'b>(
        &'a mut self,
        buf: &'b mut Vec<u8>,
        buf_start: usize
    ) -> Result<Event<'b>> {
        match read_until(&mut self.reader, b'<', buf, &mut self.buf_position) {
            Ok(None) => Ok(Event::Eof),
            Ok(_) => {
                self.last_tag_state = TagState::OpeningTagEncountered {
                    offset: buf.len(),
                    reading_state: State::Elem,
                };
                let (start, len) = (
                    buf_start
                        + if self.trim_text_start {
                            match buf.iter().skip(buf_start).position(|&b| !is_whitespace(b)) {
                                Some(start) => start,

                                // Here there is no text, just skip to the next event.
                                None => return self.read_event(buf),
                            }
                        } else {
                            0
                        },
                    if self.trim_text_end {
                        buf.iter()
                            .rposition(|&b| !is_whitespace(b))
                            .map_or_else(|| buf.len(), |p| p + 1)
                    } else {
                        buf.len()
                    },
                );
                Ok(Event::Text(BytesText::from_escaped(&buf[start..len])))
            }
            Err(e) => Err(e),
        }
    }

    /// Private function which will read until encountering the first '>' and
    /// emit an event if a new tag has been read.
    fn read_until_closing_tag<'a, 'b>(
        &'a mut self,
        buf: &'b mut Vec<u8>,
        buf_start: usize,
        reading_state: State
    ) -> Result<Event<'b>> {
        let start = if buf.len() <= buf_start {
            loop {
                match self.reader.fill_buf() {
                    Ok(n) if n.is_empty() => return Ok(Event::Eof),
                    Ok(n) => {
                        // We intentionally don't `consume()` the byte, otherwise we would have to
                        // handle things like '<>' here already.
                        break n[0];
                    }
                    Err(ref e) if e.kind() == io::ErrorKind::Interrupted => continue,
                    Err(e) => return Err(Error::Io(e)),
                }
            }
        } else {
            buf[buf_start]
        };

        if start != b'/' && start != b'!' && start != b'?' {
            match read_elem_until(&mut self.reader, b'>', buf, &mut self.buf_position, reading_state) {
                // ">" not found. Returns Eof until more data is sent.
                Ok((None, state)) => {
                    self.last_tag_state = TagState::OpeningTagEncountered {
                        offset: buf_start,
                        reading_state: state,
                    };
                    Ok(Event::Eof)
                }
                Ok(_) => {
                    self.last_tag_state = TagState::ClosingTagEncountered {
                        offset: buf.len()
                    };

                    // we already *know* that we are in this case
                    self.read_start(&buf[buf_start..])
                }
                Err(e) => Err(e),
            }
        } else {
            match read_until(&mut self.reader, b'>', buf, &mut self.buf_position) {
                // ">" not found. Returns Eof until more data is sent.
                Ok(None) => Ok(Event::Eof),
                Ok(_) => match start {
                    b'/' => {
                        self.last_tag_state = TagState::ClosingTagEncountered {
                            offset: buf.len()
                        };
                        self.read_end(&buf[buf_start..])
                    },
                    b'!' => {
                        if buf[buf_start..].starts_with(b"!--") {
                            self.last_tag_state = TagState::InsideComment {
                                offset: buf_start
                            };
                            self.read_comment(buf, buf_start)
                        } else if buf.len() >= buf_start + 8 {
                            match &buf[buf_start + 1..buf_start + 8] {
                                b"[CDATA[" => {
                                    self.last_tag_state = TagState::InsideCdata {
                                        offset: buf_start
                                    };
                                    self.read_c_data(buf, buf_start)
                                }
                                x if x.eq_ignore_ascii_case(b"DOCTYPE") => {
                                    self.last_tag_state = TagState::InsideDocType {
                                        offset: buf_start
                                    };
                                    self.read_doc_type(buf, buf_start)
                                }
                                _ => Err(Error::UnexpectedBang),
                            }
                        } else {
                            self.buf_position -= buf.len() - buf_start;
                            Err(Error::UnexpectedBang)
                        }
                    },
                    b'?' => {
                        self.last_tag_state = TagState::ClosingTagEncountered { offset: buf.len() };
                        self.read_question_mark(&buf[buf_start..])
                    },
                    _ => unreachable!(
                        "We checked that `start` must be one of [/!?], was {:?} \
                             instead.",
                        start
                    ),
                },
                Err(e) => Err(e),
            }
        }
    }

    /// reads `BytesElement` starting with a `/`,
    /// if `self.check_end_names`, checks that element matches last opened element
    /// return `End` event
    fn read_end<'a, 'b>(&'a mut self, buf: &'b [u8]) -> Result<Event<'b>> {
        // XML standard permits whitespaces after the markup name in closing tags.
        // Let's strip them from the buffer before comparing tag names.
        let name = if self.trim_markup_names_in_closing_tags {
            if let Some(pos_end_name) = buf[1..].iter().rposition(|&b| !b.is_ascii_whitespace()) {
                let (name, _) = buf[1..].split_at(pos_end_name + 1);
                name
            } else {
                &buf[1..]
            }
        } else {
            &buf[1..]
        };
        if self.check_end_names {
            let mismatch_err = |expected: &[u8], found: &[u8], buf_position: &mut usize| {
                *buf_position -= buf.len();
                Err(Error::EndEventMismatch {
                    expected: from_utf8(expected).unwrap_or("").to_owned(),
                    found: from_utf8(found).unwrap_or("").to_owned(),
                })
            };
            match self.opened_starts.pop() {
                Some(start) => {
                    if name != &self.opened_buffer[start..] {
                        let expected = &self.opened_buffer[start..];
                        mismatch_err(expected, name, &mut self.buf_position)
                    } else {
                        self.opened_buffer.truncate(start);
                        Ok(Event::End(BytesEnd::borrowed(name)))
                    }
                }
                None => mismatch_err(b"", &buf[1..], &mut self.buf_position),
            }
        } else {
            Ok(Event::End(BytesEnd::borrowed(name)))
        }
    }

    fn read_comment<'a, 'b>(
        &'a mut self,
        buf: &'b mut Vec<u8>,
        buf_start: usize,
    ) -> Result<Event<'b>> {
        while buf.len() < buf_start + 5 || !buf.ends_with(b"--") {
            buf.push(b'>');

            match read_until(&mut self.reader, b'>', buf, &mut self.buf_position) {
                Ok(None) => {
                    return Ok(Event::Eof);
                }
                Ok(_) => (),
                Err(e) => return Err(e),
            }
        }

        let len = buf.len();
        self.last_tag_state = TagState::ClosingTagEncountered { offset: len };
        if self.check_comments {
            // search if '--' not in comments
            if let Some(p) = memchr::memchr_iter(b'-', &buf[buf_start + 3..len - 2])
                .position(|p| buf[buf_start + 3 + p + 1] == b'-')
            {
                self.buf_position -= buf.len() - buf_start + p;
                return Err(Error::UnexpectedToken("--".to_string()));
            }
        }
        Ok(Event::Comment(BytesText::from_escaped(
            &buf[buf_start + 3..len - 2],
        )))
    }

    fn read_c_data<'a, 'b>(
        &'a mut self,
        buf: &'b mut Vec<u8>,
        buf_start: usize,
    ) -> Result<Event<'b>> {
        while buf.len() < 10 || !buf.ends_with(b"]]") {
            buf.push(b'>');
            match read_until(&mut self.reader, b'>', buf, &mut self.buf_position) {
                Ok(None) => {
                    return Ok(Event::Eof);
                }
                Ok(_) => (),
                Err(e) => return Err(e),
            }
        }
        self.last_tag_state = TagState::ClosingTagEncountered { offset: buf.len() };
        Ok(Event::CData(BytesText::from_plain(
                    &buf[buf_start + 8..buf.len() - 2],
        )))
    }

    fn read_doc_type<'a, 'b>(
        &'a mut self,
        buf: &'b mut Vec<u8>,
        buf_start: usize,
    ) -> Result<Event<'b>> {
        let mut count = buf.iter().skip(buf_start).filter(|&&b| b == b'<').count();
        while count > 0 {
            buf.push(b'>');
            match read_until(&mut self.reader, b'>', buf, &mut self.buf_position) {
                Ok(None) => {
                    return Ok(Event::Eof);
                }
                Ok(Some(n)) => {
                    let start = buf.len() - n;
                    count += buf.iter().skip(start).filter(|&&b| b == b'<').count();
                    count -= 1;
                }
                Err(e) => return Err(e),
            }
        }
        self.last_tag_state = TagState::ClosingTagEncountered { offset: buf.len() };
        Ok(Event::DocType(BytesText::from_escaped(
                    &buf[buf_start + 8..buf.len()],
        )))
    }

    /// reads `BytesElement` starting with a `?`,
    /// return `Decl` or `PI` event
    #[cfg(feature = "encoding")]
    fn read_question_mark<'a, 'b>(&'a mut self, buf: &'b [u8]) -> Result<Event<'b>> {
        let len = buf.len();
        if len > 2 && buf[len - 1] == b'?' {
            if len > 5 && &buf[1..4] == b"xml" && is_whitespace(buf[4]) {
                let event = BytesDecl::from_start(BytesStart::borrowed(&buf[1..len - 1], 3));
                // Try getting encoding from the declaration event
                if let Some(enc) = event.encoder() {
                    self.encoding = enc;
                    self.is_encoding_set = true;
                }
                Ok(Event::Decl(event))
            } else {
                Ok(Event::PI(BytesText::from_escaped(&buf[1..len - 1])))
            }
        } else {
            self.buf_position -= len;
            Err(Error::UnexpectedEof("XmlDecl".to_string()))
        }
    }

    /// reads `BytesElement` starting with a `?`,
    /// return `Decl` or `PI` event
    #[cfg(not(feature = "encoding"))]
    fn read_question_mark<'a, 'b>(&'a mut self, buf: &'b [u8]) -> Result<Event<'b>> {
        let len = buf.len();
        if len > 2 && buf[len - 1] == b'?' {
            if len > 5 && &buf[1..4] == b"xml" && is_whitespace(buf[4]) {
                let event = BytesDecl::from_start(BytesStart::borrowed(&buf[1..len - 1], 3));
                Ok(Event::Decl(event))
            } else {
                Ok(Event::PI(BytesText::from_escaped(&buf[1..len - 1])))
            }
        } else {
            self.buf_position -= len;
            Err(Error::UnexpectedEof("XmlDecl".to_string()))
        }
    }

    /// reads `BytesElement` starting with any character except `/`, `!` or ``?`
    /// return `Start` or `Empty` event
    fn read_start<'a, 'b>(&'a mut self, buf: &'b [u8]) -> Result<Event<'b>> {
        // TODO: do this directly when reading bufreader ...
        let len = buf.len();
        let name_end = buf.iter().position(|&b| is_whitespace(b)).unwrap_or(len);
        if let Some(&b'/') = buf.last() {
            let end = if name_end < len { name_end } else { len - 1 };
            if self.expand_empty_elements {
                self.last_tag_state = TagState::Empty;
                self.opened_starts.push(self.opened_buffer.len());
                self.opened_buffer.extend(&buf[..end]);
                Ok(Event::Start(BytesStart::borrowed(&buf[..len - 1], end)))
            } else {
                Ok(Event::Empty(BytesStart::borrowed(&buf[..len - 1], end)))
            }
        } else {
            if self.check_end_names {
                self.opened_starts.push(self.opened_buffer.len());
                self.opened_buffer.extend(&buf[..name_end]);
            }
            Ok(Event::Start(BytesStart::borrowed(buf, name_end)))
        }
    }

    /// Reads the next `Event`.
    ///
    /// This is the main entry point for reading XML `Event`s.
    ///
    /// `Event`s borrow `buf` and can be converted to own their data if needed (uses `Cow`
    /// internally).
    ///
    /// Having the possibility to control the internal buffers gives you some additional benefits
    /// such as:
    ///
    /// - Reduce the number of allocations by reusing the same buffer. For constrained systems,
    ///   you can call `buf.clear()` once you are done with processing the event (typically at the
    ///   end of your loop).
    /// - Reserve the buffer length if you know the file size (using `Vec::with_capacity`).
    ///
    /// # Examples
    ///
    /// ```
    /// use quick_xml::Reader;
    /// use quick_xml::events::Event;
    ///
    /// let xml = r#"<tag1 att1 = "test">
    ///                 <tag2><!--Test comment-->Test</tag2>
    ///                 <tag2>Test 2</tag2>
    ///             </tag1>"#;
    /// let mut reader = Reader::from_str(xml);
    /// reader.trim_text(true);
    /// let mut count = 0;
    /// let mut buf = Vec::new();
    /// let mut txt = Vec::new();
    /// loop {
    ///     match reader.read_event(&mut buf) {
    ///         Ok(Event::Start(ref e)) => count += 1,
    ///         Ok(Event::Text(e)) => txt.push(e.unescape_and_decode(&reader).expect("Error!")),
    ///         Err(e) => panic!("Error at position {}: {:?}", reader.buffer_position(), e),
    ///         Ok(Event::Eof) => break,
    ///         _ => (),
    ///     }
    ///     reader.try_clear_buffer(&mut buf);
    /// }
    /// println!("Found {} start events", count);
    /// println!("Text events: {:?}", txt);
    /// ```
    pub fn read_event<'a, 'b>(&'a mut self, buf: &'b mut Vec<u8>) -> Result<Event<'b>> {
        let event = match self.last_tag_state {
            TagState::Beginning => self.read_until_opening_tag(buf, 0),
            TagState::ClosingTagEncountered { offset } =>
                self.read_until_opening_tag(buf, offset),
            TagState::OpeningTagEncountered { offset, reading_state } =>
                self.read_until_closing_tag(buf, offset, reading_state),
            TagState::InsideComment { offset } => self.read_comment(buf, offset),
            TagState::InsideCdata { offset } => self.read_c_data(buf, offset),
            TagState::InsideDocType { offset } => self.read_doc_type(buf, offset),
            TagState::Empty => {
                self.last_tag_state = TagState::ClosingTagEncountered {
                    offset: buf.len()
                };
                let name = self
                    .opened_buffer
                    .split_off(self.opened_starts.pop().unwrap());
                Ok(Event::End(BytesEnd::owned(name)))
            },
            TagState::Exit => return Ok(Event::Eof),
        };
        match event {
            Err(_) => self.last_tag_state = TagState::Exit,
            _ => {}
        }
        event
    }

    /// Read and emit potential remaining text present after the last element.
    /// To call only after `read_event` browsed through the full XML data.
    ///
    /// The `read_event` function does not emit any event for the potential text (and most
    /// likely, whitespace) that comes after the last closing tag.
    ///
    /// If you're sure that the whole XML is now available to this parser and if you've
    /// reached Eof when calling `read_event`, you can still be notified of the remaining
    /// text through a Text event thanks to this method.
    ///
    /// ```
    /// use quick_xml::Reader;
    /// use quick_xml::events::{Event, BytesText};
    ///
    /// // Note the four spaces after the last closing tag `</tag1>`
    /// let xml = r#"<tag1 att1 = "test">
    ///                 <tag2><!--Test comment-->Test</tag2>
    ///                 <tag2>Test 2</tag2>
    ///             </tag1>    "#;
    /// let mut reader = Reader::from_str(xml);
    /// reader.trim_text(false);
    /// let mut buf = Vec::new();
    /// loop {
    ///     match reader.read_event(&mut buf) {
    ///         Ok(Event::Eof) => break,
    ///         _ => (),
    ///     }
    ///     reader.try_clear_buffer(&mut buf);
    /// }
    /// match reader.read_remaining_text(&mut buf).unwrap() {
    ///   Event::Text(content) => {
    ///     assert_eq!(&*content, b"    ");
    ///   },
    ///   _ => { panic!("Unexpected event"); },
    /// };
    /// ```
    pub fn read_remaining_text<'a, 'b>(&'a mut self, buf: &'b mut Vec<u8>) -> Result<Event<'b>> {
        match self.last_tag_state {
            TagState::ClosingTagEncountered { offset } =>
                self.read_until_end(buf, offset),
            TagState::Exit => Ok(Event::Eof),
            _ =>  Err(Error::TextNotFound),
        }
    }

    fn read_until_end<'a, 'b>(
        &'a mut self,
        buf: &'b mut Vec<u8>,
        buf_start: usize
    ) -> Result<Event<'b>> {
        let (start, len) = (
            buf_start
            + if self.trim_text_start {
                match buf.iter().skip(buf_start).position(|&b| !is_whitespace(b)) {
                    Some(start) => start,
                    None => 0,
                }
            } else {
                0
            },
            if self.trim_text_end {
                buf.iter()
                    .rposition(|&b| !is_whitespace(b))
                    .map_or_else(|| buf.len(), |p| p + 1)
            } else {
                buf.len()
            },
        );
        if len > start {
            Ok(Event::Text(BytesText::from_escaped(&buf[start..len])))
        } else {
            Ok(Event::Eof)
        }
    }

    /// Try clearing the associated buffer
    /// Returns `true` if the buffer could have been cleared
    pub fn try_clear_buffer<'a, 'b>(&'a mut self, buf: &'b mut Vec<u8>) -> bool {
        match self.last_tag_state {
            TagState::Beginning |
                TagState::Empty |
                TagState::Exit => {
                    buf.clear();
                    true
                }
            TagState::ClosingTagEncountered { offset } => {
                if offset == buf.len() {
                    buf.clear();
                    self.last_tag_state = TagState::ClosingTagEncountered { offset: 0 };
                    true
                } else {
                    false
                }
            },
            TagState::OpeningTagEncountered { offset, reading_state } => {
                if offset == buf.len() {
                    buf.clear();
                    self.last_tag_state = TagState::OpeningTagEncountered {
                        offset: 0,
                        reading_state,
                    };
                    true
                } else {
                    false
                }
            },
            TagState::InsideComment { offset } => {
                if offset == buf.len() {
                    buf.clear();
                    self.last_tag_state = TagState::InsideComment { offset: 0 };
                    true
                } else {
                    false
                }
            },
            TagState::InsideCdata { offset } => {
                if offset == buf.len() {
                    buf.clear();
                    self.last_tag_state = TagState::InsideCdata { offset: 0 };
                    true
                } else {
                    false
                }
            },
            TagState::InsideDocType { offset } => {
                if offset == buf.len() {
                    buf.clear();
                    self.last_tag_state = TagState::InsideDocType { offset: 0 };
                    true
                } else {
                    false
                }
            },
        }
    }

    /// Resolves a potentially qualified **event name** into (namespace name, local name).
    ///
    /// *Qualified* attribute names have the form `prefix:local-name` where the`prefix` is defined
    /// on any containing XML element via `xmlns:prefix="the:namespace:uri"`. The namespace prefix
    /// can be defined on the same element as the attribute in question.
    ///
    /// *Unqualified* event inherits the current *default namespace*.
    #[inline]
    pub fn event_namespace<'a, 'b, 'c>(
        &'a self,
        qname: &'b [u8],
        namespace_buffer: &'c [u8],
    ) -> (Option<&'c [u8]>, &'b [u8]) {
        self.ns_buffer
            .resolve_namespace(qname, namespace_buffer, true)
    }

    /// Resolves a potentially qualified **attribute name** into (namespace name, local name).
    ///
    /// *Qualified* attribute names have the form `prefix:local-name` where the`prefix` is defined
    /// on any containing XML element via `xmlns:prefix="the:namespace:uri"`. The namespace prefix
    /// can be defined on the same element as the attribute in question.
    ///
    /// *Unqualified* attribute names do *not* inherit the current *default namespace*.
    #[inline]
    pub fn attribute_namespace<'a, 'b, 'c>(
        &'a self,
        qname: &'b [u8],
        namespace_buffer: &'c [u8],
    ) -> (Option<&'c [u8]>, &'b [u8]) {
        self.ns_buffer
            .resolve_namespace(qname, namespace_buffer, false)
    }

    /// Reads the next event and resolves its namespace (if applicable).
    ///
    /// # Examples
    ///
    /// ```
    /// use std::str::from_utf8;
    /// use quick_xml::Reader;
    /// use quick_xml::events::Event;
    ///
    /// let xml = r#"<x:tag1 xmlns:x="www.xxxx" xmlns:y="www.yyyy" att1 = "test">
    ///                 <y:tag2><!--Test comment-->Test</y:tag2>
    ///                 <y:tag2>Test 2</y:tag2>
    ///             </x:tag1>"#;
    /// let mut reader = Reader::from_str(xml);
    /// reader.trim_text(true);
    /// let mut count = 0;
    /// let mut buf = Vec::new();
    /// let mut ns_buf = Vec::new();
    /// let mut txt = Vec::new();
    /// loop {
    ///     match reader.read_namespaced_event(&mut buf, &mut ns_buf) {
    ///         Ok((ref ns, Event::Start(ref e))) => {
    ///             count += 1;
    ///             match (*ns, e.local_name()) {
    ///                 (Some(b"www.xxxx"), b"tag1") => (),
    ///                 (Some(b"www.yyyy"), b"tag2") => (),
    ///                 (ns, n) => panic!("Namespace and local name mismatch"),
    ///             }
    ///             println!("Resolved namespace: {:?}", ns.and_then(|ns| from_utf8(ns).ok()));
    ///         }
    ///         Ok((_, Event::Text(e))) => {
    ///             txt.push(e.unescape_and_decode(&reader).expect("Error!"))
    ///         },
    ///         Err(e) => panic!("Error at position {}: {:?}", reader.buffer_position(), e),
    ///         Ok((_, Event::Eof)) => break,
    ///         _ => (),
    ///     }
    ///     buf.clear();
    /// }
    /// println!("Found {} start events", count);
    /// println!("Text events: {:?}", txt);
    /// ```
    pub fn read_namespaced_event<'a, 'b, 'c>(
        &'a mut self,
        buf: &'b mut Vec<u8>,
        namespace_buffer: &'c mut Vec<u8>,
    ) -> Result<(Option<&'c [u8]>, Event<'b>)> {
        self.ns_buffer.pop_empty_namespaces(namespace_buffer);
        match self.read_event(buf) {
            Ok(Event::Eof) => Ok((None, Event::Eof)),
            Ok(Event::Start(e)) => {
                self.ns_buffer.push_new_namespaces(&e, namespace_buffer);
                Ok((
                    self.ns_buffer
                        .find_namespace_value(e.name(), &**namespace_buffer),
                    Event::Start(e),
                ))
            }
            Ok(Event::Empty(e)) => {
                // For empty elements we need to 'artificially' keep the namespace scope on the
                // stack until the next `next()` call occurs.
                // Otherwise the caller has no chance to use `resolve` in the context of the
                // namespace declarations that are 'in scope' for the empty element alone.
                // Ex: <img rdf:nodeID="abc" xmlns:rdf="urn:the-rdf-uri" />
                self.ns_buffer.push_new_namespaces(&e, namespace_buffer);
                // notify next `read_namespaced_event()` invocation that it needs to pop this
                // namespace scope
                self.ns_buffer.pending_pop = true;
                Ok((
                    self.ns_buffer
                        .find_namespace_value(e.name(), &**namespace_buffer),
                    Event::Empty(e),
                ))
            }
            Ok(Event::End(e)) => {
                // notify next `read_namespaced_event()` invocation that it needs to pop this
                // namespace scope
                self.ns_buffer.pending_pop = true;
                Ok((
                    self.ns_buffer
                        .find_namespace_value(e.name(), &**namespace_buffer),
                    Event::End(e),
                ))
            }
            Ok(e) => Ok((None, e)),
            Err(e) => Err(e),
        }
    }

    /// Returns the `Reader`s encoding.
    ///
    /// The used encoding may change after parsing the XML declaration.
    ///
    /// This encoding will be used by [`decode`].
    ///
    /// [`decode`]: #method.decode
    #[cfg(feature = "encoding")]
    pub fn encoding(&self) -> &'static Encoding {
        self.encoding
    }

    /// Decodes a slice using the encoding specified in the XML declaration.
    ///
    /// Decode `bytes` with BOM sniffing and with malformed sequences replaced with the
    /// `U+FFFD REPLACEMENT CHARACTER`.
    ///
    /// If no encoding is specified, defaults to UTF-8.
    #[inline]
    #[cfg(feature = "encoding")]
    pub fn decode<'b, 'c>(&'b self, bytes: &'c [u8]) -> Cow<'c, str> {
        self.encoding.decode(bytes).0
    }

    /// Decodes a UTF8 slice without BOM (Byte order mark) regardless of XML declaration.
    ///
    /// Decode `bytes` without BOM and with malformed sequences replaced with the
    /// `U+FFFD REPLACEMENT CHARACTER`.
    ///
    /// # Note
    ///
    /// If you instead want to use XML declared encoding, use the `encoding` feature
    #[inline]
    #[cfg(not(feature = "encoding"))]
    pub fn decode_without_bom<'c>(&self, bytes: &'c [u8]) -> Result<&'c str> {
        if bytes.starts_with(b"\xEF\xBB\xBF") {
            from_utf8(&bytes[3..]).map_err(Error::Utf8)
        } else {
            from_utf8(bytes).map_err(Error::Utf8)
        }
    }

    /// Decodes a slice using without BOM (Byte order mark) the encoding specified in the XML declaration.
    ///
    /// Decode `bytes` without BOM and with malformed sequences replaced with the
    /// `U+FFFD REPLACEMENT CHARACTER`.
    ///
    /// If no encoding is specified, defaults to UTF-8.
    #[inline]
    #[cfg(feature = "encoding")]
    pub fn decode_without_bom<'b, 'c>(&'b mut self, mut bytes: &'c [u8]) -> Cow<'c, str> {
        if self.is_encoding_set {
            return self.encoding.decode_with_bom_removal(bytes).0;
        }
        if bytes.starts_with(b"\xEF\xBB\xBF") {
            self.is_encoding_set = true;
            bytes = &bytes[3..];
        } else if bytes.starts_with(b"\xFF\xFE") {
            self.is_encoding_set = true;
            self.encoding = UTF_16LE;
            bytes = &bytes[2..];
        } else if bytes.starts_with(b"\xFE\xFF") {
            self.is_encoding_set = true;
            self.encoding = UTF_16BE;
            bytes = &bytes[3..];
        };
        self.encoding.decode_without_bom_handling(bytes).0
    }

    /// Decodes a UTF8 slice regardless of XML declaration.
    ///
    /// Decode `bytes` with BOM sniffing and with malformed sequences replaced with the
    /// `U+FFFD REPLACEMENT CHARACTER`.
    ///
    /// # Note
    ///
    /// If you instead want to use XML declared encoding, use the `encoding` feature
    #[inline]
    #[cfg(not(feature = "encoding"))]
    pub fn decode<'c>(&self, bytes: &'c [u8]) -> Result<&'c str> {
        from_utf8(bytes).map_err(Error::Utf8)
    }

    /// Get utf8 decoder
    #[cfg(feature = "encoding")]
    pub fn decoder(&self) -> Decoder {
        Decoder {
            encoding: self.encoding,
        }
    }

    /// Get utf8 decoder
    #[cfg(not(feature = "encoding"))]
    pub fn decoder(&self) -> Decoder {
        Decoder
    }

    /// Reads until end element is found
    ///
    /// Manages nested cases where parent and child elements have the same name
    pub fn read_to_end<K: AsRef<[u8]>>(&mut self, end: K, buf: &mut Vec<u8>) -> Result<()> {
        let mut depth = 0;
        let end = end.as_ref();
        loop {
            match self.read_event(buf) {
                Ok(Event::End(ref e)) if e.name() == end => {
                    if depth == 0 {
                        return Ok(());
                    }
                    depth -= 1;
                }
                Ok(Event::Start(ref e)) if e.name() == end => depth += 1,
                Err(e) => return Err(e),
                Ok(Event::Eof) => {
                    return Err(Error::UnexpectedEof(format!("</{:?}>", from_utf8(end))));
                }
                _ => (),
            }
            self.try_clear_buffer(buf);
        }
    }

    /// Reads optional text between start and end tags.
    ///
    /// If the next event is a [`Text`] event, returns the decoded and unescaped content as a
    /// `String`. If the next event is an [`End`] event, returns the empty string. In all other
    /// cases, returns an error.
    ///
    /// Any text will be decoded using the XML encoding specified in the XML declaration (or UTF-8
    /// if none is specified).
    ///
    /// # Examples
    ///
    /// ```
    ///   // XXX TODO
    ///   // use quick_xml::Reader;
    ///   // use quick_xml::events::Event;
    ///   //
    ///   // let mut buf = Vec::new();
    ///   // let mut xml = Reader::from_reader(b"
    ///   //     <a>&lt;b&gt;</a>
    ///   //     <a></a>
    ///   // " as &[u8]);
    ///   // xml.trim_text(true);
    ///   //
    ///   // let expected = ["<b>", ""];
    ///   // for &content in expected.iter() {
    ///   //     match xml.read_event(&mut buf) {
    ///   //         Ok(Event::Start(ref e)) => {
    ///   //             assert_eq!(&xml.read_text(e.name(), &mut buf).unwrap(), content);
    ///   //         },
    ///   //         e => panic!("Expecting Start event, found {:?}", e),
    ///   //     }
    ///   // }
    /// ```
    ///
    /// [`Text`]: events/enum.Event.html#variant.Text
    /// [`End`]: events/enum.Event.html#variant.End
    pub fn read_text<K: AsRef<[u8]>>(&mut self, end: K, buf: &mut Vec<u8>) -> Result<String> {
        let s = match self.read_event(buf) {
            Ok(Event::Text(e)) => e.unescape_and_decode(self),
            Ok(Event::End(ref e)) if e.name() == end.as_ref() => return Ok("".to_string()),
            Err(e) => return Err(e),
            Ok(Event::Eof) => return Err(Error::UnexpectedEof("Text".to_string())),
            _ => return Err(Error::TextNotFound),
        };
        self.read_to_end(end, buf)?;
        s
    }

    /// Consumes `Reader` returning the underlying reader
    ///
    /// Can be used to compute line and column of a parsing error position
    ///
    /// # Examples
    ///
    /// ```
    /// use std::{str, io::Cursor};
    /// use quick_xml::Reader;
    /// use quick_xml::events::Event;
    ///
    /// let xml = r#"<tag1 att1 = "test">
    ///                 <tag2><!--Test comment-->Test</tag2>
    ///                 <tag3>Test 2</tag3>
    ///             </tag1>"#;
    /// let mut reader = Reader::from_reader(Cursor::new(xml.as_bytes()));
    /// let mut buf = Vec::new();
    ///
    /// fn into_line_and_column(reader: Reader<Cursor<&[u8]>>) -> (usize, usize) {
    ///     let end_pos = reader.buffer_position();
    ///     let mut cursor = reader.into_underlying_reader();
    ///     let s = String::from_utf8(cursor.into_inner()[0..end_pos].to_owned())
    ///         .expect("can't make a string");
    ///     let mut line = 1;
    ///     let mut column = 0;
    ///     for c in s.chars() {
    ///         if c == '\n' {
    ///             line += 1;
    ///             column = 0;
    ///         } else {
    ///             column += 1;
    ///         }
    ///     }
    ///     (line, column)
    /// }
    ///
    /// loop {
    ///     match reader.read_event(&mut buf) {
    ///         Ok(Event::Start(ref e)) => match e.name() {
    ///             b"tag1" | b"tag2" => (),
    ///             tag => {
    ///                 assert_eq!(b"tag3", tag);
    ///                 assert_eq!((3, 22), into_line_and_column(reader));
    ///                 break;
    ///             }
    ///         },
    ///         Ok(Event::Eof) => unreachable!(),
    ///         _ => (),
    ///     }
    ///     reader.try_clear_buffer(&mut buf);
    /// }
    /// ```
    pub fn into_underlying_reader(self) -> B {
        self.reader
    }
}

impl Reader<BufReader<File>> {
    /// Creates an XML reader from a file path.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Reader<BufReader<File>>> {
        let file = File::open(path).map_err(Error::Io)?;
        let reader = BufReader::new(file);
        Ok(Reader::from_reader(reader))
    }
}

impl<'a> Reader<&'a [u8]> {
    /// Creates an XML reader from a string slice.
    pub fn from_str(s: &'a str) -> Reader<&'a [u8]> {
        Reader::from_reader(s.as_bytes())
    }
}

/// read until `byte` is found and returns the amount of bytes read until
/// it has been found.
///
/// Returns Ok(None) if EOF has been reached without finding anything.
#[inline]
fn read_until<R: BufRead>(
    r: &mut R,
    byte: u8,
    buf: &mut Vec<u8>,
    position: &mut usize,
) -> Result<Option<usize>> {
    let mut read = 0;
    let mut done = false;
    while !done {
        let used = {
            let available = match r.fill_buf() {
                Ok(n) if n.is_empty() => {
                    *position += read;
                    println!("POS: {}", position);
                    return Ok(None);
                },
                Ok(n) => n,
                Err(ref e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => {
                    *position += read;
                    return Err(Error::Io(e));
                }
            };

            match memchr::memchr(byte, available) {
                Some(i) => {
                    buf.extend_from_slice(&available[..i]);
                    done = true;
                    i + 1
                }
                None => {
                    buf.extend_from_slice(available);
                    available.len()
                }
            }
        };
        r.consume(used);
        read += used;
    }
    *position += read;
    Ok(Some(read))
}

#[derive(Clone, Copy)]
enum State {
    /// The initial state (inside element, but outside of attribute value)
    Elem,
    /// Inside a single-quoted attribute value
    SingleQ,
    /// Inside a double-quoted attribute value
    DoubleQ,
}

/// Derived from `read_until`, but modified to handle XML attributes using a minimal state machine.
/// [W3C Extensible Markup Language (XML) 1.1 (2006)](https://www.w3.org/TR/xml11)
///
/// Attribute values are defined as follows:
/// ```plain
/// AttValue := '"' (([^<&"]) | Reference)* '"'
///           | "'" (([^<&']) | Reference)* "'"
/// ```
/// (`Reference` is something like `&quot;`, but we don't care about escaped characters at this
/// level)
#[inline]
fn read_elem_until<R: BufRead>(
    r: &mut R,
    end_byte: u8,
    buf: &mut Vec<u8>,
    position: &mut usize,
    initial_state: State
) -> Result<(Option<usize>, State)> {
    let mut state = initial_state;
    let mut read = 0;
    let mut done = false;
    while !done {
        let used = {
            let available = match r.fill_buf() {
                Ok(n) if n.is_empty() => {
                    *position += read;
                    return Ok((None, state));
                },
                Ok(n) => n,
                Err(ref e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => {
                    *position += read;
                    return Err(Error::Io(e));
                }
            };

            let mut memiter = memchr::memchr3_iter(end_byte, b'\'', b'"', available);
            let used: usize;
            loop {
                match memiter.next() {
                    Some(i) => {
                        state = match (state, available[i]) {
                            (State::Elem, b) if b == end_byte => {
                                // only allowed to match `end_byte` while we are in state `Elem`
                                buf.extend_from_slice(&available[..i]);
                                done = true;
                                used = i + 1;
                                break;
                            }
                            (State::Elem, b'\'') => State::SingleQ,
                            (State::Elem, b'\"') => State::DoubleQ,

                            // the only end_byte that gets us out if the same character
                            (State::SingleQ, b'\'') | (State::DoubleQ, b'\"') => State::Elem,

                            // all other bytes: no state change
                            _ => state,
                        };
                    }
                    None => {
                        buf.extend_from_slice(available);
                        used = available.len();
                        break;
                    }
                }
            }
            used
        };
        r.consume(used);
        read += used;
    }
    *position += read;
    Ok((Some(read), state))
}

/// A function to check whether the byte is a whitespace (blank, new line, carriage return or tab)
#[inline]
pub(crate) fn is_whitespace(b: u8) -> bool {
    match b {
        b' ' | b'\r' | b'\n' | b'\t' => true,
        _ => false,
    }
}

/// A namespace declaration. Can either bind a namespace to a prefix or define the current default
/// namespace.
#[derive(Debug, Clone)]
struct Namespace {
    /// Index of the namespace in the buffer
    start: usize,
    /// Length of the prefix
    /// * if bigger than start, then binds this namespace to the corresponding slice.
    /// * else defines the current default namespace.
    prefix_len: usize,
    /// The namespace name (the URI) of this namespace declaration.
    ///
    /// The XML standard specifies that an empty namespace value 'removes' a namespace declaration
    /// for the extent of its scope. For prefix declarations that's not very interesting, but it is
    /// vital for default namespace declarations. With `xmlns=""` you can revert back to the default
    /// behaviour of leaving unqualified element names unqualified.
    value_len: usize,
    /// Level of nesting at which this namespace was declared. The declaring element is included,
    /// i.e., a declaration on the document root has `level = 1`.
    /// This is used to pop the namespace when the element gets closed.
    level: i32,
}

impl Namespace {
    /// Gets the value slice out of namespace buffer
    ///
    /// Returns `None` if `value_len == 0`
    #[inline]
    fn opt_value<'a, 'b>(&'a self, ns_buffer: &'b [u8]) -> Option<&'b [u8]> {
        if self.value_len == 0 {
            None
        } else {
            let start = self.start + self.prefix_len;
            Some(&ns_buffer[start..start + self.value_len])
        }
    }

    /// Check if the namespace matches the potentially qualified name
    #[inline]
    fn is_match(&self, ns_buffer: &[u8], qname: &[u8]) -> bool {
        if self.prefix_len == 0 {
            !qname.contains(&b':')
        } else {
            qname.get(self.prefix_len).map_or(false, |n| *n == b':')
                && qname.starts_with(&ns_buffer[self.start..self.start + self.prefix_len])
        }
    }
}

/// A namespace management buffer.
///
/// Holds all internal logic to push/pop namespaces with their levels.
#[derive(Debug, Default, Clone)]
struct NamespaceBufferIndex {
    /// a buffer of namespace ranges
    slices: Vec<Namespace>,
    /// The number of open tags at the moment. We need to keep track of this to know which namespace
    /// declarations to remove when we encounter an `End` event.
    nesting_level: i32,
    /// For `Empty` events keep the 'scope' of the element on the stack artificially. That way, the
    /// consumer has a chance to use `resolve` in the context of the empty element. We perform the
    /// pop as the first operation in the next `next()` call.
    pending_pop: bool,
}

impl NamespaceBufferIndex {
    #[inline]
    fn find_namespace_value<'a, 'b, 'c>(
        &'a self,
        element_name: &'b [u8],
        buffer: &'c [u8],
    ) -> Option<&'c [u8]> {
        self.slices
            .iter()
            .rfind(|n| n.is_match(buffer, element_name))
            .and_then(|n| n.opt_value(buffer))
    }

    fn pop_empty_namespaces(&mut self, buffer: &mut Vec<u8>) {
        if !self.pending_pop {
            return;
        }
        self.pending_pop = false;
        self.nesting_level -= 1;
        let current_level = self.nesting_level;
        // from the back (most deeply nested scope), look for the first scope that is still valid
        match self.slices.iter().rposition(|n| n.level <= current_level) {
            // none of the namespaces are valid, remove all of them
            None => {
                buffer.clear();
                self.slices.clear();
            }
            // drop all namespaces past the last valid namespace
            Some(last_valid_pos) => {
                if let Some(len) = self.slices.get(last_valid_pos + 1).map(|n| n.start) {
                    buffer.truncate(len);
                    self.slices.truncate(last_valid_pos + 1);
                }
            }
        }
    }

    fn push_new_namespaces(&mut self, e: &BytesStart, buffer: &mut Vec<u8>) {
        self.nesting_level += 1;
        let level = self.nesting_level;
        // adds new namespaces for attributes starting with 'xmlns:' and for the 'xmlns'
        // (default namespace) attribute.
        for a in e.attributes().with_checks(false) {
            if let Ok(Attribute { key: k, value: v }) = a {
                if k.starts_with(b"xmlns") {
                    match k.get(5) {
                        None => {
                            let start = buffer.len();
                            buffer.extend_from_slice(&*v);
                            self.slices.push(Namespace {
                                start,
                                prefix_len: 0,
                                value_len: v.len(),
                                level,
                            });
                        }
                        Some(&b':') => {
                            let start = buffer.len();
                            buffer.extend_from_slice(&k[6..]);
                            buffer.extend_from_slice(&*v);
                            self.slices.push(Namespace {
                                start,
                                prefix_len: k.len() - 6,
                                value_len: v.len(),
                                level,
                            });
                        }
                        _ => break,
                    }
                }
            } else {
                break;
            }
        }
    }

    /// Resolves a potentially qualified **attribute name** into (namespace name, local name).
    ///
    /// *Qualified* attribute names have the form `prefix:local-name` where the`prefix` is defined
    /// on any containing XML element via `xmlns:prefix="the:namespace:uri"`. The namespace prefix
    /// can be defined on the same element as the attribute in question.
    ///
    /// *Unqualified* attribute names do *not* inherit the current *default namespace*.
    #[inline]
    fn resolve_namespace<'a, 'b, 'c>(
        &'a self,
        qname: &'b [u8],
        buffer: &'c [u8],
        use_default: bool,
    ) -> (Option<&'c [u8]>, &'b [u8]) {
        self.slices
            .iter()
            .rfind(|n| n.is_match(buffer, qname))
            .map_or((None, qname), |n| {
                let len = n.prefix_len;
                if len > 0 {
                    (n.opt_value(buffer), &qname[len + 1..])
                } else if use_default {
                    (n.opt_value(buffer), qname)
                } else {
                    (None, qname)
                }
            })
    }
}

/// Utf8 Decoder
#[cfg(not(feature = "encoding"))]
#[derive(Clone, Copy)]
pub struct Decoder;

/// Utf8 Decoder
#[cfg(feature = "encoding")]
#[derive(Clone, Copy)]
pub struct Decoder {
    encoding: &'static Encoding,
}

impl Decoder {
    #[cfg(not(feature = "encoding"))]
    pub fn decode<'c>(&self, bytes: &'c [u8]) -> Result<&'c str> {
        from_utf8(bytes).map_err(Error::Utf8)
    }

    #[cfg(feature = "encoding")]
    pub fn decode<'c>(&self, bytes: &'c [u8]) -> Cow<'c, str> {
        self.encoding.decode(bytes).0
    }
}
