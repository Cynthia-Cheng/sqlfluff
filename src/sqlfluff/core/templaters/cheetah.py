import logging
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union, cast
from Cheetah.Compiler import Compiler
from Cheetah.Parser import _HighLevelParser
import regex
from dataclasses import dataclass, field
from Cheetah.Template import Template
from sqlfluff.core import FluffConfig
from sqlfluff.core.errors import SQLTemplaterError
from sqlfluff.core.templaters.base import (
    RawFileSlice,
    RawTemplater,
    TemplatedFile,
    TemplatedFileSlice,
    large_file_check,
)


templater_logger = logging.getLogger("sqlfluff.templater")


class CheetahTemplater(RawTemplater):
    name = "cheetah"

    def __init__(self, override_context=None, **kwargs):
        self.default_context = dict(test_value="__test__")
        self.override_context = override_context or {}

    @large_file_check
    def process(
            self, *, in_str: str, fname: str, config=None, formatter=None
    ) -> Tuple[TemplatedFile, List[SQLTemplaterError]]:
        render_func = self.construct_render_func(fname=fname, config=config)

        try:
            rendered_sql = render_func(in_str)
            # TODO 解析出所有的变量。。。
        except Exception as err:
            templater_error = SQLTemplaterError(
                "Failed to parse Cheetah syntax. " + str(err)
            )
            # TODO 尽可能解析出error中的行号。。。
            raise templater_error

        try:
            # Slice the file once rendered.
            raw_sliced, sliced_file, out_str = self.slice_file(
                in_str,
                render_func=render_func,
                config=config,
            )
            return (
                TemplatedFile(
                    source_str=in_str,
                    templated_str=out_str,
                    fname=fname,
                    sliced_file=sliced_file,
                    raw_sliced=raw_sliced,
                ),
                [],
            )
        except (Exception, TypeError) as err:
            templater_logger.info("Unrecoverable Cheetah Error: %s", err, exc_info=True)
            raise SQLTemplaterError(
                "Unrecoverable failure in Cheetah templating: {}. ".format(err),
                # We don't have actual line number information, but specify
                # line 1 so users can ignore with "noqa" if they want. (The
                # default is line 0, which can't be ignored because it's not
                # a valid line number.)
                line_no=1,
                line_pos=1,
            )

    def slice_file(
            self, raw_str: str, render_func: Callable[[str], str], config=None, **kwargs
    ) -> Tuple[List[RawFileSlice], List[TemplatedFileSlice], str]:
        """Slice the file to determine regions where we can fix.

        Args:
            raw_str (str): The raw string to be sliced.
            render_func (Callable[[str], str]): The rendering function to be used.
            config (optional): Optional configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[List[RawFileSlice], List[TemplatedFileSlice], str]:
                A tuple containing a list of raw file slices, a list of
                templated file slices, and the templated string.
        """

        templater_logger.info("Slicing File Template")
        templater_logger.debug("    Raw String: %r", raw_str[:80])
        analyzer = CheetahAnalyzer(raw_str)
        tracer = analyzer.analyze(render_func)
        trace = tracer.trace()
        # output_list is used to check if the mapping result is correct， do not uncomment in the prod env
        # output_list = []
        # for slice in trace.sliced_file:
        #     output = (raw_str[slice.source_slice.start:slice.source_slice.stop],
        #               trace.templated_str[slice.templated_slice.start:slice.templated_slice.stop])
        #     output_list.append(output)

        return trace.raw_sliced, trace.sliced_file, trace.templated_str

    def construct_render_func(self, fname=None, config=None) -> Callable[[str], str]:
        """Builds and returns objects needed to create and run templates.

        Args:
            fname (Optional[str]): The name of the file.
            config (Optional[dict]): The configuration settings.

        Returns:
            render_func (Callable[[str], str]): A callable function
                that is used to instantiate templates.
        """

        def render_func(in_str: str) -> str:
            """Used by CheetahTracer to instantiate templates.

            This function is a closure capturing internal state from process().
            Note that creating templates involves quite a bit of state known to
            _this_ function but not to CheetahTracer.
            """
            try:
                search_list = self._get_search_list_from_config(config)
                template = Template(in_str, searchList=search_list)
            except Exception as err:  # pragma: no cover
                # NOTE: If the template fails to parse, then this clause
                # will be triggered. However in normal that should never
                # happen because the template should already have been
                # validated by the point this is called. Typically that
                # happens when searching for undefined variables.
                raise SQLTemplaterError(
                    f"Late failure to parse cheetah template: {err}.",
                    line_no=getattr(err, "lineno", None),
                )
            return str(template)

        return render_func

    def _get_search_list_from_config(self, config: FluffConfig) -> Optional[dict]:
        """Get the search_list from the provided config object.

        Args:
            config (FluffConfig): The config object to search for the search_list
                section.

        Returns:
            Optional[List[str]]: The dict of search_list if found, None otherwise.
        """
        if config:
            search_list = config.get_section(("core", self.name, "variable"))
            return search_list
        return None

    def calculate_coverage(self, templated_files: Union[TemplatedFile, List[TemplatedFile]], src_str: str):
        if not isinstance(templated_files, list):
            templated_file = templated_files
            covered_positions = {tfs.source_slice.start for tfs in templated_file.sliced_file}
        else:
            templated_file = templated_files[0]
            covered_positions = {tfs.source_slice.start for tf in templated_files for tfs in tf.sliced_file}

        # 未覆盖区域
        uncovered_slices = [
            (raw_slice.source_idx, templated_file.raw_sliced[idx + 1].source_idx)
            for idx, raw_slice in enumerate(templated_file.raw_sliced[:-1])
            if raw_slice.source_idx not in covered_positions
        ]
        uncovered_len = sum(end - start for start, end in uncovered_slices)
        # 字符覆盖率
        cover_rate = (len(src_str) - uncovered_len) / len(src_str)

        # 行覆盖情况
        line_count, _ = templated_file.get_line_pos_of_char_pos(len(src_str) - 1)
        coverage = [1] * (line_count + 1)
        for start, end in uncovered_slices:
            start_line, _ = templated_file.get_line_pos_of_char_pos(start)
            end_line, _ = templated_file.get_line_pos_of_char_pos(end)
            coverage[start_line:end_line] = [0] * (end_line - start_line)


        print(cover_rate)


class CheetahTrace(NamedTuple):
    """Returned by CheetahTracer.trace()."""

    # Template output
    templated_str: str
    # Raw (i.e. before rendering) Cheetah template sliced into tokens
    raw_sliced: List[RawFileSlice]
    # Rendered Cheetah template (i.e. output) mapped back to rwa_str source
    sliced_file: List[TemplatedFileSlice]


@dataclass
class RawSliceInfo:
    """CheetahTracer-specific info about each RawFileSlice."""

    unique_alternate_id: Optional[str]
    alternate_code: Optional[str]
    next_slice_indices: List[int] = field(default_factory=list)


class CheetahTracer:
    """Records execution path of a Cheetah template."""

    def __init__(
            self,
            raw_str: str,
            raw_sliced: List[RawFileSlice],
            raw_slice_info: Dict[RawFileSlice, RawSliceInfo],
            sliced_file: List[TemplatedFileSlice],
            render_func: Callable[[str], str],
    ):
        # Input
        self.raw_str = raw_str
        self.raw_sliced = raw_sliced
        self.raw_slice_info = raw_slice_info
        self.sliced_file = sliced_file
        self.render_func = render_func

        # Internal bookkeeping
        self.program_counter: int = 0
        self.source_idx: int = 0

    def trace(self) -> CheetahTrace:
        """Executes raw_str. Returns template output and trace."""
        trace_template_str = "".join(
            (
                cast(str, self.raw_slice_info[rs].alternate_code)
                if self.raw_slice_info[rs].alternate_code is not None
                else rs.raw
            )
            for rs in self.raw_sliced
        )
        trace_template_output = self.render_func(trace_template_str)
        # Split output by section. Each section has two possible formats.
        trace_entries: List[regex.Match[str]] = list(
            regex.finditer(r"\x1F", trace_template_output)
        )
        # If the file has no templated entries, we should just iterate
        # through the raw slices to add all the placeholders.
        if not trace_entries:
            for raw_idx, _ in enumerate(self.raw_sliced):
                self.record_trace(0, raw_idx)

        for match_idx, match in enumerate(trace_entries):
            pos1 = match.span()[0]
            try:
                pos2 = trace_entries[match_idx + 1].span()[0]
            except IndexError:
                pos2 = len(trace_template_output)
            p = trace_template_output[pos1 + 1: pos2]
            m_id = regex.match(r"^([0-9a-f]+)(_(\d+))?", p)
            if not m_id:
                raise ValueError(  # pragma: no cover
                    "Internal error. Trace template output does not match expected "
                    "format."
                )
            if m_id.group(3):
                # E.g. "00000000000000000000000000000001_83". The number after
                # "_" is the length (in characters) of a corresponding literal
                # in raw_str.
                alt_id, slice_length = m_id.group(1), int(m_id.group(3))
            else:
                # alternate_code = f"\x1F{unique_alternate_id} \x1E{raw}\x1E"
                # E.g. "00000000000000000000000000000002 \x1Ea < 10\x1E". The characters
                # between the \x1E are executable code from raw_str.
                wrap_match = list(regex.finditer(r"\x1E", p))
                if len(wrap_match) != 2:
                    raise ValueError(
                        "Internal error. Trace template output does not match expected "
                        "format."
                    )
                alt_id, slice_length = m_id.group(0), wrap_match[1].start() - wrap_match[0].start() - 1

            target_slice_idx = self.find_slice_index(alt_id)
            self.move_to_slice(target_slice_idx, slice_length)

        templated_str = self.render_func(self.raw_str)
        return CheetahTrace(templated_str, self.raw_sliced, self.sliced_file)

    def find_slice_index(self, slice_identifier: Union[int, str]) -> int:
        """Given a slice identifier, return its index.

        A slice identifier is a string like 00000000000000000000000000000002.
        """
        raw_slices_search_result = [
            idx
            for idx, rs in enumerate(self.raw_sliced)
            if self.raw_slice_info[rs].unique_alternate_id == slice_identifier
        ]
        if len(raw_slices_search_result) != 1:
            raise ValueError(  # pragma: no cover
                f"Internal error. Unable to locate slice for {slice_identifier}."
            )
        return raw_slices_search_result[0]

    def move_to_slice(
            self,
            target_slice_idx: int,
            target_slice_length: int,
    ) -> Dict[int, List[int]]:
        """Given a template location, walk execution to that point.

        This updates the internal `program_counter` to the appropriate
        location.

        Returns:
            :obj:`dict`: For each step in the template, a :obj:`list` of
                which steps are accessible. In many cases each step will
                only have one accessible next step (the following one),
                however for branches in the program there may be more than
                one.
        """
        step_candidates = {}
        while self.program_counter < len(self.raw_sliced):
            self.record_trace(
                target_slice_length if self.program_counter == target_slice_idx else 0
            )  # self.program_counter == target_slice_idx 计数器还未到达target，证明前面都是为模版，长度为0，将其加入到slice_file
            current_raw_slice = self.raw_sliced[self.program_counter]
            if self.program_counter == target_slice_idx:
                # Reached the target slice. Go to next location and stop.
                self.program_counter += 1
                break

            # Choose the next step.
            # We could simply go to the next slice (sequential execution).
            candidates = [self.program_counter + 1]
            # If we have other options, consider those.
            candidates.extend(
                filter(
                    # They're a valid possibility if
                    # they don't take us past the target.
                    lambda idx: idx <= target_slice_idx,
                    self.raw_slice_info[current_raw_slice].next_slice_indices,
                )
            )
            # Choose the candidate that takes us closest to the target.
            candidates.sort(key=lambda c: abs(target_slice_idx - c))
            # Save all the candidates for each step so we can return them later.
            step_candidates[self.program_counter] = candidates
            # Step forward to the best step found.
            # TODO end_for或者end_while的next_slice_indices可能有多个吗 ？
            if self.raw_slice_info[current_raw_slice].next_slice_indices and len(
                    self.raw_slice_info[current_raw_slice].next_slice_indices) > 1:
                raise ValueError(  # pragma: no cover
                    "uncover situation: next_slice 长度超过1."
                )
            if (current_raw_slice.tag in ("end_for", "end_while")
                    and self.raw_slice_info[current_raw_slice].next_slice_indices  # 在def里面next_slice_indices为空, 不追踪
                    and self.raw_slice_info[current_raw_slice].next_slice_indices[
                        0] < target_slice_idx < self.program_counter):
                self.program_counter = self.raw_slice_info[current_raw_slice].next_slice_indices[0]
            else:
                self.program_counter = candidates[0]

        # Return the candidates at each step.
        return step_candidates

    def record_trace(
            self,
            target_slice_length: int,
            slice_idx: Optional[int] = None,
            slice_type: Optional[str] = None,
    ) -> None:
        """Add the specified (default: current) location to the trace.

        Args:
            target_slice_length (int): The length of the target slice.
            slice_idx (Optional[int], optional): The index of the slice.
            Defaults to None.
            slice_type (Optional[str], optional): The type of the slice.
            Defaults to None.
        """
        if slice_idx is None:
            slice_idx = self.program_counter
        if slice_type is None:
            slice_type = self.raw_sliced[slice_idx].slice_type
        self.sliced_file.append(
            TemplatedFileSlice(
                slice_type,
                slice(
                    self.raw_sliced[slice_idx].source_idx,
                    (
                        self.raw_sliced[slice_idx + 1].source_idx
                        if slice_idx + 1 < len(self.raw_sliced)
                        else len(self.raw_str)
                    ),
                ),
                slice(self.source_idx, self.source_idx + target_slice_length),
            )
        )
        if target_slice_length:
            self.source_idx += target_slice_length


@dataclass(frozen=True)
class CheetahTagConfiguration:
    """Provides information about a Cheetah tag and how it affects CheetahAnalyzer behavior.

    Attributes:
        block_type (str): The block type that the Cheetah tag maps to; eventually stored
            in TemplatedFileSlice.slice_type and RawFileSlice.slice_type.
        block_tracking (bool): Whether the Cheetah tag should be traced by CheetahTracer.
            If True, the Cheetah tag will be treated as a conditional block similar to a
            "for/endfor" or "if/else/endif" block, and CheetahTracer will track potential
            execution path through the block.
        block_may_loop (bool): Whether the Cheetah tag begins a block that might loop,
            similar to a "for" tag.  If True, CheetahTracer will track the execution path
            through the block and record a potential backward jump to the loop
            beginning.
    """

    block_type: str
    block_tracking: bool = False
    block_may_loop: bool = False


class CheetahAnalyzer:
    def __init__(self, raw_str: str) -> None:
        # Input
        self.raw_str: str = raw_str

        # Output
        self.raw_sliced: List[RawFileSlice] = []
        self.raw_slice_info: Dict[RawFileSlice, RawSliceInfo] = {}
        self.sliced_file: List[TemplatedFileSlice] = []

        # Internal bookkeeping
        self.slice_id: int = 0
        self.is_inside_def: bool = False
        self.stack: List[int] = []
        self.untrack_stack: List[int] = []
        self.idx_raw: int = 0

    __known_tag_configurations = {
        # Conditional blocks: "if/elif/else/endif" blocks
        "if": CheetahTagConfiguration(
            block_type="block_start",
            block_tracking=True,
        ),
        "else": CheetahTagConfiguration(
            block_type="block_mid",
            block_tracking=True,
        ),
        "elif": CheetahTagConfiguration(
            block_type="block_mid",
            block_tracking=True,
        ),
        "end_if": CheetahTagConfiguration(
            block_type="block_end",
            block_tracking=True,
        ),
        "end_for": CheetahTagConfiguration(
            block_type="block_end",
            block_tracking=True,
        ),
        "end_while": CheetahTagConfiguration(
            block_type="block_end",
            block_tracking=True,
        ),
        "end_def": CheetahTagConfiguration(
            block_type="block_end",
            block_tracking=True,
        ),
        # Conditional blocks: "for" loops
        "for": CheetahTagConfiguration(
            block_type="block_start",
            block_tracking=True,
            block_may_loop=True,
        ),
        "while": CheetahTagConfiguration(
            block_type="block_start",
            block_tracking=True,
            block_may_loop=True,
        ),
        "end": CheetahTagConfiguration(
            block_type="block_end",
            block_tracking=True,
        ),
    }

    def analyze(self, render_func: Callable[[str], str]):
        block_idx = 0
        for raw, elem_type in CheetahLexer(self.raw_str).lex():
            if elem_type == "literal":
                self.track_literal(raw, block_idx)
                continue
            block_type = "templated"
            block_tag = None
            raw_slice_info: RawSliceInfo = self.make_raw_slice_info(None, None)
            if elem_type == "variable" or elem_type == "directive-echo":
                raw_slice_info = self.track_templated(raw)
            elif elem_type.startswith("directive") or elem_type in ("comment", "multiline_comment"):
                raw_slice_info = self.track_directive(raw, elem_type)

            # 处理block元素
            if elem_type in self.block_types:
                block_type = self.block_types[elem_type]
                block_tag = elem_type.split("-")[1]
                if block_type == "block_start":
                    block_idx += 1
            # 更新block_tag
            if block_type == "block_end":
                # 找最近的block_start
                open_tag = next((tag for tag in ["while", "for", "if", "def"] if tag in raw), None)
                block_tag = f"end_{open_tag}"
            # 处理其他，比如#set，或余下
            self.raw_sliced.append(
                RawFileSlice(
                    raw,
                    block_type,
                    self.idx_raw,
                    block_idx,
                    block_tag,
                )
            )
            self.raw_slice_info[self.raw_sliced[-1]] = raw_slice_info
            slice_idx = len(self.raw_sliced) - 1
            self.idx_raw += len(raw)
            if block_type == "block_end":
                block_idx += 1
            if block_type.startswith("block") or (block_tag and block_tag in ("continue", "break")):
                self.track_block_end(block_type, str(block_tag))
                self.update_next_slice_indices(slice_idx, block_type, str(block_tag))
                self.update_inside_def(slice_idx, block_type, raw)

        return CheetahTracer(
            self.raw_str,
            self.raw_sliced,
            self.raw_slice_info,
            self.sliced_file,
            render_func,
        )

    def track_templated(self, raw: str) -> RawSliceInfo:
        """Compute tracking info for Cheetah templated region, e.g. $foo.

        """
        unique_alternate_id = self.next_slice_id()
        # Here, we still need to evaluate the original tag contents, e.g. in
        # case it has intentional side effects, but also return a slice ID
        # for tracking.
        alternate_code = f"\x1F{unique_alternate_id} \x1E{raw}\x1E"
        return self.make_raw_slice_info(unique_alternate_id, alternate_code)

    def track_directive(self, raw: str, elem_type: str) -> RawSliceInfo:
        """"
        Example:
                00000000000000000000000000000000_1#def date_filter($delta=0, $extra_delta=0, $reload_delta=0, $dt_key='dt')
                    #if $isSLICERELOAD
                        $dt_key = '2024'
                    #end if
                #end def

                select c from tb
                    where ${date_filter(0, 0, 1)}

                The 00000000000000000000000000000000_1 before #def will cause an extra newline to be rendered for ${date_filter(0, 0, 1)}. !!!!!
                , so add a newline before it to prevent tracer mapping errors.
        行内元素包裹
        Example:
            FROM #if 'exp_key' in $type
            tb
        这个if是行内元素，但是没有被#包裹，任何出现在该行内的元素都会被视为if的一部分，会干扰tracer

        Example:
            -- #if True:
            -- tb
            -- #end if
        #if True:# anything #end if# 其中end会识别不出来

        非block元素特殊处理，alternate_code
        Example:
            #if True
                #set $sourcedb_app = "app_peisongqc_test"
                #set $sourcedb_mart = "mart_peisong_test"
            #else
                #set $sourcedb_app = "app_peisongqc"
                #set $sourcedb_mart = "mart_peisong"
            #end if
            $sourcedb_app
        tracer在走到$sourcedb_app时由于if block块内没有渲染出任何东西，会忽略掉两个#set行，所以在其后加上全局id，将其当作渲染出来的一个标志
        """
        # 另起一行，确保编码后的code可以独占一行，前面的换行会被tracer忽略，并让cheetah可以吸收末尾的换行
        unique_alternate_id = self.next_slice_id()
        if elem_type in self.block_types:
            alternate_code = f"\n{raw.strip()}\n"
        else:
            alternate_code = f"\n{raw.strip()}\n\x1F{unique_alternate_id}_0"
        return self.make_raw_slice_info(unique_alternate_id, alternate_code)

    def track_literal(self, raw: str, block_idx: int) -> None:
        """Set up tracking for a Cheetah literal."""
        self.raw_sliced.append(
            RawFileSlice(
                raw,
                "literal",
                self.idx_raw,
                block_idx,
            )
        )
        # Replace literal text with a unique ID.
        self.raw_slice_info[self.raw_sliced[-1]] = self.slice_info_for_literal(
            len(raw), ""
        )
        self.idx_raw += len(raw)

    def track_block_end(self, block_type: str, tag_name: str) -> None:
        """On ending a 'for' or 'if' block, set up tracking.

        Args:
            block_type (str): The type of block ('block_start', 'block_mid',
                'block_end').
            tag_name (str): The name of the tag ('for', 'if', or other configured tag).
        """
        if (
                block_type == "block_end"
                and self._get_tag_configuration(tag_name).block_tracking
        ):
            # Replace RawSliceInfo for this slice with one that has alternate ID
            # and code for tracking. This ensures, for instance, that if a file
            # ends with "# end if  (with no newline following), that we still
            # generate a TemplateSliceInfo for it.
            unique_alternate_id = self.next_slice_id()
            end_fragment = self.raw_sliced[-1].raw.strip()
            alternate_code = f"\n{end_fragment}\n\x1F{unique_alternate_id}_0"
            self.raw_slice_info[self.raw_sliced[-1]] = self.make_raw_slice_info(
                unique_alternate_id, alternate_code
            )

    # update inside_def
    def update_inside_def(self, slice_idx: int, block_type: str, raw: str) -> None:
        # TODO 除了def还有其他元素，比如block
        if block_type == "block_start" and self.raw_sliced[slice_idx].tag in ["def"]:
            self.untrack_stack.append(slice_idx)
            self.is_inside_def = True
        elif block_type == "block_end" and 'def' in raw:
            # 是 end def
            self.untrack_stack.pop()
            if not self.untrack_stack and not raw.endswith("\n"):
                # 离开def，更新raw_slice_info的code
                self.raw_slice_info[self.raw_sliced[slice_idx]].alternate_code = f"#{raw.strip().strip('#')}#"
        if self.untrack_stack:
            self.is_inside_def = True
        else:
            self.is_inside_def = False

    def update_next_slice_indices(
            self, slice_idx: int, block_type: str, tag_name: str
    ) -> None:
        """Based on block, update conditional jump info."""
        if (
                block_type == "block_start"
                and self._get_tag_configuration(tag_name).block_tracking
        ):
            self.stack.append(slice_idx)
            return None
        elif not self.stack:
            return None

        _idx = self.stack[-1]
        _raw_slice = self.raw_sliced[_idx]
        _slice_info = self.raw_slice_info[_raw_slice]
        if (
                block_type == "block_mid"
                and self._get_tag_configuration(tag_name).block_tracking
        ):
            # Record potential forward jump over this block.
            _slice_info.next_slice_indices.append(slice_idx)
            self.stack.pop()
            self.stack.append(slice_idx)
        elif (
                block_type == "block_end"
                and self._get_tag_configuration(tag_name).block_tracking
        ):
            if not self.is_inside_def:
                # Record potential forward jump over this block.
                _slice_info.next_slice_indices.append(slice_idx)
                self.stack.pop()
                if _raw_slice.slice_type == "block_start":
                    assert _raw_slice.tag
                    if self._get_tag_configuration(_raw_slice.tag).block_may_loop:
                        # Record potential backward jump to the loop beginning.
                        self.raw_slice_info[
                            self.raw_sliced[slice_idx]
                        ].next_slice_indices.append(_idx + 1)
                        # 处理loop中的break
                        # 从self.raw_sliced[_idx + 1, slice_idx)找所有的break slice，构建break_indices->slice_idx
                        for break_slice_idx in range(_idx + 1, slice_idx):
                            if self.raw_sliced[break_slice_idx].tag == "break":
                                self.raw_slice_info[self.raw_sliced[break_slice_idx]].next_slice_indices.append(
                                    slice_idx)
        # 处理loop中的continue
        elif block_type == "templated" and tag_name == "continue":
            if not self.is_inside_def:
                # 在stack里面找while或者for，找continue跳转的slice
                loop_start_id = next(slice_idx for slice_idx in reversed(self.stack) if
                                     self.raw_sliced[slice_idx].tag in ("while", "for"))
                self.raw_slice_info[self.raw_sliced[slice_idx]].next_slice_indices.append(loop_start_id + 1)

    def slice_info_for_literal(self, length: int, prefix: str = "") -> RawSliceInfo:
        unique_alternate_id = self.next_slice_id()
        alternate_code = f"\x1F{prefix}{unique_alternate_id}_{length}"
        return self.make_raw_slice_info(
            unique_alternate_id, alternate_code
        )

    @classmethod
    def _get_tag_configuration(cls, tag: str) -> CheetahTagConfiguration:
        """Return information about the behaviors of a tag."""
        # Ideally, we should have a known configuration for this Cheetah tag.  Derived
        # classes can override this method to provide additional information about the
        # tags they know about.
        known_cfg = cls.__known_tag_configurations.get(tag, None)
        if known_cfg:
            return known_cfg

        # If we don't have a firm configuration for this tag that is most likely
        # provided by a Cheetah extension, we'll try to make some guesses about it based
        # on some heuristics.  But there's a decent chance we'll get this wrong, and
        # the user should instead consider overriding this method in a derived class to
        # handle their tag types.
        if tag.startswith("end"):
            return CheetahTagConfiguration(
                block_type="block_end",
            )
        elif tag.startswith("el"):
            # else, elif
            return CheetahTagConfiguration(
                block_type="block_mid",
            )
        return CheetahTagConfiguration(
            block_type="block_start",
        )

    def next_slice_id(self) -> str:
        """Returns a new, unique slice ID."""
        result = "{0:#0{1}x}".format(self.slice_id, 34)[2:]
        self.slice_id += 1
        return result

    def make_raw_slice_info(
            self,
            unique_alternate_id: Optional[str],
            alternate_code: Optional[str],
    ) -> RawSliceInfo:
        """Create RawSliceInfo as given, or "empty" if in def block."""
        if not self.is_inside_def:
            return RawSliceInfo(unique_alternate_id, alternate_code, [])
        else:
            # 如果在def里面，则不做任何编码，def block内部不追踪
            return RawSliceInfo(None, None, [])

    block_types = {
        # "variable": "templated",
        "directive-for": "block_start",
        "directive-if": "block_start",
        "directive-def": "block_start",
        "directive-while": "block_start",
        "directive-end": "block_end",
        "directive-else": "block_mid",
        "directive-elif": "block_mid",
        "directive-continue": "templated",
        "directive-break": "templated",
    }

assignOp = "="
augAssignOps = (
    "+=",
    "-=",
    "/=",
    "*=",
    "**=",
    "^=",
    "%=",
    ">>=",
    "<<=",
    "&=",
    "|=",
)
assignmentOps = (assignOp,) + augAssignOps
SET_LOCAL = 0
SET_GLOBAL = 1
SET_MODULE = 2


class ParseError(ValueError):
    def __init__(self, stream, msg='Invalid Syntax', extMsg='',
                 lineno=None, col=None):
        self.stream = stream
        if stream.pos() >= len(stream):
            stream.setPos(len(stream) - 1)
        self.msg = msg
        self.extMsg = extMsg
        self.lineno = lineno
        self.col = col

    def __str__(self):
        return self.report()

    def report(self):
        stream = self.stream
        if stream.filename():
            f = " in file %s" % stream.filename()
        else:
            f = ''
        report = ''
        if self.lineno:
            lineno = self.lineno
            row, col, line = (lineno, (self.col or 0),
                              self.stream.splitlines()[lineno - 1])
        else:
            row, col, line = self.stream.getRowColLine()

        # get the surrounding lines
        lines = stream.splitlines()
        prevLines = []  # (rowNum, content)
        for i in range(1, 4):
            if row - 1 - i <= 0:
                break
            prevLines.append((row - i, lines[row - 1 - i]))

        nextLines = []  # (rowNum, content)
        for i in range(1, 4):
            if not row - 1 + i < len(lines):
                break
            nextLines.append((row + i, lines[row - 1 + i]))
        nextLines.reverse()

        # print the main message
        report += "\n\n%s\n" % self.msg
        report += "Line %i, column %i%s\n\n" % (row, col, f)
        report += 'Line|Cheetah Code\n'
        report += '----|-----------------------------' \
                  '--------------------------------\n'
        while prevLines:
            lineInfo = prevLines.pop()
            report += "%(row)-4d|%(line)s\n" \
                      % {'row': lineInfo[0], 'line': lineInfo[1]}
        report += "%(row)-4d|%(line)s\n" % {'row': row, 'line': line}
        report += ' ' * 5 + ' ' * (col - 1) + "^\n"  # noqa: E226,E501 missing whitespace around operator

        while nextLines:
            lineInfo = nextLines.pop()
            report += "%(row)-4d|%(line)s\n" \
                      % {'row': lineInfo[0], 'line': lineInfo[1]}
        # add the extra msg
        if self.extMsg:
            report += self.extMsg + '\n'

        return report


class CustomCompiler(Compiler):
    def __init__(self, template_str):
        super().__init__()
        self.template_str = template_str

    def addChunk(self, chunk):
        pass

    def indent(self):
        pass

    def dedent(self):
        pass

    def commitStrConst(self):
        pass

    def startMethodDef(self, methodName, argsList, parserComment):
        pass

    def closeDef(self):
        pass

    def handleWSBeforeDirective(self):
        pass


class CheetahLexer(_HighLevelParser):
    def __init__(self, template_str):
        super().__init__(template_str, compiler=CustomCompiler(template_str))
        self.isSingleLineDef = False
        self.isAbsorbLeadingIndent = False

    def lex(self) -> List[Tuple[str, str]]:
        template_parts = []
        while not self.atEnd():
            part_start = self._pos
            type = "literal"
            self.isAbsorbLeadingIndent = False
            if self.matchCommentStartToken():
                self.isAbsorbLeadingIndent = self.isLineClearToStartToken() and len(template_parts) > 0
                absorbStart = self.findBOL()
                type = "comment"
                self.eatComment()
            elif self.matchMultiLineCommentStartToken():
                self.isAbsorbLeadingIndent = self.isLineClearToStartToken() and len(template_parts) > 0
                absorbStart = self.findBOL()
                type = "multiline_comment"
                self.eatMultiLineComment()
            elif self.matchVariablePlaceholderStart():
                type = "variable"
                self.eatPlaceholder()
            elif self.matchExpressionPlaceholderStart():
                type = "variable"
                self.eatPlaceholder()
            elif self.matchDirective():
                self.isAbsorbLeadingIndent = self.isLineClearToStartToken() and len(template_parts) > 0
                absorbStart = self.findBOL()
                type = "directive-" + self.eatDirective()  # done todo 其他指令测试、macro
                if self.isSingleLineDef:
                    type = "directive-inline-def"
                    self.isSingleLineDef = False
            elif self.matchPSPStartToken():
                type = "psp"  # todo
                self.eatPSP()
            elif self.matchEOLSlurpToken():
                type = "slurp"  # todo
                self.eatEOLSlurpToken()
            else:
                self.eatPlainText()  # done
            # Handling leading indentation For directives
            if self.isAbsorbLeadingIndent:
                last_str, last_type = template_parts[-1]
                if len(last_str.rstrip(self._src[absorbStart:part_start])) == 0:
                    template_parts[-1] = (self._src[absorbStart: self._pos], type)
                else:
                    template_parts[-1] = (
                        last_str.rstrip(self._src[absorbStart:part_start]),
                        last_type,
                    )
                    template_parts.append((self._src[absorbStart: self._pos], type))
            else:
                template_parts.append((self._src[part_start: self._pos], type))
        # Security check: Ensure consistency after lexical splitting
        reconstructed_template = ''.join(part for part, _ in template_parts)
        assert reconstructed_template == self._src, "The sliced template is inconsistent with the original template."
        return template_parts

    def eatComment(self):
        isLineClearToStartToken = self.isLineClearToStartToken()
        self.getCommentStartToken()
        self.readToEOL(gobble=isLineClearToStartToken)

    def eatMultiLineComment(self):
        isLineClearToStartToken = self.isLineClearToStartToken()
        endOfFirstLine = self.findEOL()

        self.getMultiLineCommentStartToken()
        endPos = startPos = self.pos()
        level = 1
        while True:
            endPos = self.pos()
            if self.atEnd():
                break
            if self.matchMultiLineCommentStartToken():
                self.getMultiLineCommentStartToken()
                level += 1
            elif self.matchMultiLineCommentEndToken():
                self.getMultiLineCommentEndToken()
                level -= 1
            if not level:
                break
            self.advance()
        self.readTo(endPos, start=startPos)

        if not self.atEnd():
            self.getMultiLineCommentEndToken()

        if (not self.atEnd()) and \
                self.setting('gobbleWhitespaceAroundMultiLineComments'):
            restOfLine = self[self.pos():self.findEOL()]
            if not restOfLine.strip():  # WS only to EOL
                self.readToEOL(gobble=isLineClearToStartToken)

    def _eatSingleLineDef(self, directiveName, methodName, argsList, startPos, endPos):
        super()._eatSingleLineDef(directiveName, methodName, argsList, startPos, endPos)
        self.isSingleLineDef = True

    def eatPlainText(self):
        startPos = self.pos()
        match = None
        while not self.atEnd():
            match = self.matchTopLevelToken()
            if match:
                break
            else:
                self.advance()
        strConst = self.readTo(self.pos(), start=startPos)
        strConst = self._unescapeCheetahVars(strConst)
        strConst = self._unescapeDirectives(strConst)
        return strConst

    def _eatRestOfDirectiveTag(self, isLineClearToStartToken,
                               endOfFirstLinePos):
        foundComment = False
        if self.matchCommentStartToken():
            pos = self.pos()
            self.advance()
            if not self.matchDirective():
                self.setPos(pos)
                foundComment = True
                # Note: directive后发现注释，行前缩进不吸收
                self.isAbsorbLeadingIndent = False
                self.eatComment()  # this won't gobble the EOL
            else:
                self.setPos(pos)

        if not foundComment and self.matchDirectiveEndToken():
            self.getDirectiveEndToken()
        elif isLineClearToStartToken and (not self.atEnd()) \
                and self.peek() in '\r\n':
            # still gobble the EOL if a comment was found.
            self.readToEOL(gobble=True)

    def eatSet(self):
        # filtered
        isLineClearToStartToken = self.isLineClearToStartToken()
        endOfFirstLine = self.findEOL()
        self.getDirectiveStartToken()
        self.advance(3)
        self.getWhiteSpace()
        style = SET_LOCAL
        if self.startswith('local'):
            self.getIdentifier()
            self.getWhiteSpace()
        elif self.startswith('global'):
            self.getIdentifier()
            self.getWhiteSpace()
            style = SET_GLOBAL
        elif self.startswith('module'):
            self.getIdentifier()
            self.getWhiteSpace()
            style = SET_MODULE

        startPos = self.pos()
        LVALUE = self.getExpression(pyTokensToBreakAt=assignmentOps,
                                    useNameMapper=False).strip()
        OP = self.getAssignmentOperator()
        RVALUE = self.getExpression()
        expr = LVALUE + ' ' + OP + ' ' + RVALUE.strip()

        expr = self._applyExpressionFilters(expr, 'set', startPos=startPos)
        self._eatRestOfDirectiveTag(isLineClearToStartToken, endOfFirstLine)

        # used for 'set global'
        class Components:
            pass

        exprComponents = Components()
        exprComponents.LVALUE = LVALUE
        exprComponents.OP = OP
        exprComponents.RVALUE = RVALUE

    def eatPlaceholder(self):
        self.getPlaceholder(
            allowCacheTokens=True, returnEverything=True)

    def _eatMultiLineDef(self, directiveName, methodName, argsList, startPos,
                         isLineClearToStartToken=False):
        # filtered in calling method
        self.getExpression()  # slurp up any garbage left at the end
        signature = self[startPos:self.pos()]
        endOfFirstLinePos = self.findEOL()
        self._eatRestOfDirectiveTag(isLineClearToStartToken, endOfFirstLinePos)
        signature = ' '.join([line.strip() for line in signature.splitlines()])
        parserComment = ('## CHEETAH: generated from ' + signature
                         + ' at line %s, col %s' % self.getRowCol(startPos)
                         + '.')

        isNestedDef = (self.setting('allowNestedDefScopes')
                       and len([name for name in self._openDirectivesStack
                                if name == 'def']) > 1)
        # if directiveName == 'block' or (
        # directiveName == 'def' and not isNestedDef):
        # self._compiler.startMethodDef(methodName, argsList, parserComment)
        # else:  # closure
        #     self._useSearchList_orig = self.setting('useSearchList')
        #     self.setSetting('useSearchList', False)
        # self._compiler.addClosure(methodName, argsList, parserComment)

        return methodName

    def handleEndDef(self):
        isNestedDef = (self.setting('allowNestedDefScopes')
                       and [name for name in self._openDirectivesStack
                            if name == 'def'])
        # if not isNestedDef:
        # self._compiler.closeDef()
        # else:
        # @@TR: temporary hack of useSearchList
        # self.setSetting('useSearchList', self._useSearchList_orig)
        # self._compiler.commitStrConst()
        # self._compiler.dedent()

    def eatIf(self):
        # filtered
        isLineClearToStartToken = self.isLineClearToStartToken()
        endOfFirstLine = self.findEOL()
        lineCol = self.getRowCol()
        self.getDirectiveStartToken()
        startPos = self.pos()

        expressionParts = self.getExpressionParts(pyTokensToBreakAt=[':'])
        expr = ''.join(expressionParts).strip()
        expr = self._applyExpressionFilters(expr, 'if', startPos=startPos)

        isTernaryExpr = ('then' in expressionParts
                         and 'else' in expressionParts)
        if isTernaryExpr:
            conditionExpr = []
            trueExpr = []
            falseExpr = []
            currentExpr = conditionExpr
            for part in expressionParts:
                if part.strip() == 'then':
                    currentExpr = trueExpr
                elif part.strip() == 'else':
                    currentExpr = falseExpr
                else:
                    currentExpr.append(part)

            conditionExpr = ''.join(conditionExpr)
            trueExpr = ''.join(trueExpr)
            falseExpr = ''.join(falseExpr)
            self._eatRestOfDirectiveTag(isLineClearToStartToken,
                                        endOfFirstLine)
            # self._compiler.addTernaryExpr(conditionExpr, trueExpr,
            #                               falseExpr, lineCol=lineCol)
        elif self.matchColonForSingleLineShortFormDirective():
            self.advance()  # skip over :
            # self._compiler.addIf(expr, lineCol=lineCol)
            self.getWhiteSpace(max=1)
            self.parse(breakPoint=self.findEOL(gobble=True))
            # self._compiler.commitStrConst()
            # self._compiler.dedent()
        else:
            if self.peek() == ':':
                self.advance()
            self.getWhiteSpace()
            self._eatRestOfDirectiveTag(isLineClearToStartToken,
                                        endOfFirstLine)
            self.pushToOpenDirectivesStack('if')
            # self._compiler.addIf(expr, lineCol=lineCol)

    def eatDirective(self):
        directiveName = self.matchDirective()
        self._filterDisabledDirectives(directiveName)

        for callback in self.setting('preparseDirectiveHooks'):
            callback(parser=self, directiveName=directiveName)

        # subclasses can override the default behaviours here by providing an
        # eater method in self._directiveNamesAndParsers[directiveName]
        directiveParser = self._directiveNamesAndParsers.get(directiveName)
        if directiveParser:
            directiveParser()
        elif directiveName in self._simpleIndentingDirectives:
            self.eatSimpleIndentingDirective(directiveName)
        elif directiveName in self._simpleExprDirectives:
            if directiveName in ('silent', 'echo'):
                includeDirectiveNameInExpr = False
            else:
                includeDirectiveNameInExpr = True
            self.eatSimpleExprDirective(
                directiveName,
                includeDirectiveNameInExpr=includeDirectiveNameInExpr)
        return directiveName

    def eatAttr(self):
        # filtered
        isLineClearToStartToken = self.isLineClearToStartToken()
        endOfFirstLinePos = self.findEOL()
        startPos = self.pos()
        self.getDirectiveStartToken()
        self.advance(len('attr'))
        self.getWhiteSpace()
        startPos = self.pos()
        if self.matchCheetahVarStart():
            self.getCheetahVarStartToken()
        attribName = self.getIdentifier()
        self.getWhiteSpace()
        self.getAssignmentOperator()
        expr = self.getExpression()
        expr = self._applyExpressionFilters(expr, 'attr', startPos=startPos)
        # self._compiler.addAttribute(attribName, expr)
        self._eatRestOfDirectiveTag(isLineClearToStartToken, endOfFirstLinePos)

    def eatSimpleIndentingDirective(self, directiveName, includeDirectiveNameInExpr=False):
        # filtered
        isLineClearToStartToken = self.isLineClearToStartToken()
        endOfFirstLinePos = self.findEOL()
        lineCol = self.getRowCol()
        self.getDirectiveStartToken()
        if directiveName not in \
                'else elif for while try except finally'.split():
            self.advance(len(directiveName))
        startPos = self.pos()

        self.getWhiteSpace()

        expr = self.getExpression(pyTokensToBreakAt=[':'])
        expr = self._applyExpressionFilters(expr, directiveName,
                                            startPos=startPos)
        if self.matchColonForSingleLineShortFormDirective():
            self.advance()  # skip over :
            self.getWhiteSpace(max=1)
            self.parse(breakPoint=self.findEOL(gobble=True))
        else:
            if self.peek() == ':':
                self.advance()
            self.getWhiteSpace()
            self._eatRestOfDirectiveTag(isLineClearToStartToken,
                                        endOfFirstLinePos)
            if directiveName in self._closeableDirectives:
                self.pushToOpenDirectivesStack(directiveName)

    def eatPSP(self):
        # filtered
        self._filterDisabledDirectives(directiveName='psp')
        self.getPSPStartToken()
        endToken = self.setting('PSPEndToken')
        startPos = self.pos()
        while not self.atEnd():
            if self.peek() == endToken[0]:
                if self.matchPSPEndToken():
                    break
            self.advance()
        pspString = self.readTo(self.pos(), start=startPos).strip()
        self._applyExpressionFilters(pspString, 'psp',
                                     startPos=startPos)
        self.getPSPEndToken()

    def eatEndDirective(self):
        isLineClearToStartToken = self.isLineClearToStartToken()
        self.getDirectiveStartToken()
        self.advance(3)  # to end of 'end'
        self.getWhiteSpace()
        pos = self.pos()
        directiveName = False
        for key in self._endDirectiveNamesAndHandlers.keys():
            if self.find(key, pos) == pos:
                directiveName = key
                break
        if not directiveName:
            raise ParseError(self, msg='Invalid end directive')

        endOfFirstLinePos = self.findEOL()
        self.getExpression()  # eat in any extra comment-like crap
        self._eatRestOfDirectiveTag(isLineClearToStartToken, endOfFirstLinePos)
        if directiveName in self._closeableDirectives:
            self.popFromOpenDirectivesStack(directiveName)

        if self._endDirectiveNamesAndHandlers.get(directiveName):
            handler = self._endDirectiveNamesAndHandlers[directiveName]
            handler()
