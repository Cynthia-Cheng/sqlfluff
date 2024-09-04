import pytest

import logging
from sqlfluff.core.templaters import TemplatedFile
from sqlfluff.core.templaters.cheetah import CheetahTemplater

CHEETAH_STRING = (
    """
#set $table = 'my_table'

SELECT *
FROM $table
"""
)


@pytest.mark.parametrize(
    "instr, expected_outstr",
    [
        (
            CHEETAH_STRING,
            "\n\nSELECT *\nFROM my_table\n",
        )
    ],
    ids=["simple"],
)
def test__templater_cheetah(instr, expected_outstr):
    """Test jinja templating and the treatment of whitespace."""
    t = CheetahTemplater()
    outstr, _ = t.process(
        in_str=instr, fname="test")
    assert str(outstr) == expected_outstr


@pytest.mark.parametrize(
    "raw_file,override_context,result,templater_class",
    [
        ("", None, [], CheetahTemplater),
        (
            "foo",
            None,
            [("literal", slice(0, 3, None), slice(0, 3, None))],
            CheetahTemplater,
        ),
        ("""
#set $table = 'my_table'

SELECT *
FROM $table
""",
         None,
         [("literal", slice(0, 1, None), slice(0, 1, None)),
          ("templated", slice(1, 26, None), slice(1, 1, None)),
          ("literal", slice(26, 41, None), slice(1, 16, None)),
          ("templated", slice(41, 47, None), slice(16, 24, None)),
          ("literal", slice(47, 48, None), slice(24, 25, None))],
         CheetahTemplater,),
        ("""
#set $items = ["apple", "banana", "cherry", "date"]

SELECT
#for $item in $items
    $item,
#end for
FROM my_table
""",
         None,
         [("literal", slice(0, 1, None), slice(0, 1, None)),
          ("templated", slice(1, 53, None), slice(1, 1, None)),
          ("literal", slice(53, 61, None), slice(1, 9, None)),
          ("block_start", slice(61, 82, None), slice(9, 9, None)),
          ("literal", slice(82, 86, None), slice(9, 13, None)),
          ("templated", slice(86, 91, None), slice(13, 18, None)),
          ("literal", slice(91, 93, None), slice(18, 20, None)),
          ("block_end", slice(93, 102, None), slice(20, 20, None)),
          ("literal", slice(82, 86, None), slice(20, 24, None)),
          ("templated", slice(86, 91, None), slice(24, 30, None)),
          ("literal", slice(91, 93, None), slice(30, 32, None)),
          ("block_end", slice(93, 102, None), slice(32, 32, None)),
          ("literal", slice(82, 86, None), slice(32, 36, None)),
          ("templated", slice(86, 91, None), slice(36, 42, None)),
          ("literal", slice(91, 93, None), slice(42, 44, None)),
          ("block_end", slice(93, 102, None), slice(44, 44, None)),
          ("literal", slice(82, 86, None), slice(44, 48, None)),
          ("templated", slice(86, 91, None), slice(48, 52, None)),
          ("literal", slice(91, 93, None), slice(52, 54, None)),
          ("block_end", slice(93, 102, None), slice(54, 54, None)),
          ("literal", slice(102, 116, None), slice(54, 68, None))],
         CheetahTemplater),
        ("""
#set $table = 'my_table'
#set $isENTER = 'true'

SELECT id,
#if $isENTER == 'true'
concat(c1,c2,c3) as new_col
#end if
FROM $table
""",
         None,
         [("literal", slice(0, 1, None), slice(0, 1, None)),
          ("templated", slice(1, 26, None), slice(1, 1, None)),
          ("templated", slice(26, 49, None), slice(1, 1, None)),
          ("literal", slice(49, 61, None), slice(1, 13, None)),
          ("block_start", slice(61, 84, None), slice(13, 13, None)),
          ("literal", slice(84, 112, None), slice(13, 41, None)),
          ("block_end", slice(112, 120, None), slice(41, 41, None)),
          ("literal", slice(120, 125, None), slice(41, 46, None)),
          ("templated", slice(125, 131, None), slice(46, 54, None)),
          ("literal", slice(131, 132, None), slice(54, 55, None))],
         CheetahTemplater),
        ("""
#*
这是一个带有for和if嵌套的Cheetah模板SQL示例。
*#

#set $table_name = "employees"
#set $columns = ["id", "name", "age", "department"]
#set $conditions = [
    {"column": "age", "operator": ">", "value": 30},
    {"column": "department", "operator": "=", "value": "Sales"}
]

SELECT
    #for $index, $column in enumerate($columns)
        $column#if $index < len($columns) - 1#, #end if#
    #end for
FROM
    $table_name
WHERE
    #for $index, $condition in enumerate($conditions)
        $condition.column $condition.operator $condition.value#if $index < len($conditions) - 1# AND #end if#
    #end for
;
""",
         None,
         [("literal", slice(0, 1, None), slice(0, 1, None)), ("templated", slice(1, 38, None), slice(1, 1, None)),
          ("literal", slice(38, 39, None), slice(1, 2, None)), ("templated", slice(39, 70, None), slice(2, 2, None)),
          ("templated", slice(70, 122, None), slice(2, 2, None)),
          ("templated", slice(122, 262, None), slice(2, 2, None)),
          ("literal", slice(262, 270, None), slice(2, 10, None)),
          ("block_start", slice(270, 318, None), slice(10, 10, None)),
          ("literal", slice(318, 326, None), slice(10, 18, None)),
          ("templated", slice(326, 333, None), slice(18, 20, None)),
          ("block_start", slice(333, 364, None), slice(20, 20, None)),
          ("literal", slice(364, 366, None), slice(20, 22, None)),
          ("block_end", slice(366, 374, None), slice(22, 22, None)),
          ("literal", slice(374, 375, None), slice(22, 23, None)),
          ("block_end", slice(375, 388, None), slice(23, 23, None)),
          ("literal", slice(318, 326, None), slice(23, 31, None)),
          ("templated", slice(326, 333, None), slice(31, 35, None)),
          ("block_start", slice(333, 364, None), slice(35, 35, None)),
          ("literal", slice(364, 366, None), slice(35, 37, None)),
          ("block_end", slice(366, 374, None), slice(37, 37, None)),
          ("literal", slice(374, 375, None), slice(37, 38, None)),
          ("block_end", slice(375, 388, None), slice(38, 38, None)),
          ("literal", slice(318, 326, None), slice(38, 46, None)),
          ("templated", slice(326, 333, None), slice(46, 49, None)),
          ("block_start", slice(333, 364, None), slice(49, 49, None)),
          ("literal", slice(364, 366, None), slice(49, 51, None)),
          ("block_end", slice(366, 374, None), slice(51, 51, None)),
          ("literal", slice(374, 375, None), slice(51, 52, None)),
          ("block_end", slice(375, 388, None), slice(52, 52, None)),
          ("literal", slice(318, 326, None), slice(52, 60, None)),
          ("templated", slice(326, 333, None), slice(60, 70, None)),
          ("block_start", slice(333, 364, None), slice(70, 70, None)),
          ("block_end", slice(366, 374, None), slice(70, 70, None)),
          ("literal", slice(374, 375, None), slice(70, 71, None)),
          ("block_end", slice(375, 388, None), slice(71, 71, None)),
          ("literal", slice(388, 397, None), slice(71, 80, None)),
          ("templated", slice(397, 408, None), slice(80, 89, None)),
          ("literal", slice(408, 415, None), slice(89, 96, None)),
          ("block_start", slice(415, 469, None), slice(96, 96, None)),
          ("literal", slice(469, 477, None), slice(96, 104, None)),
          ("templated", slice(477, 494, None), slice(104, 107, None)),
          ("literal", slice(494, 495, None), slice(107, 108, None)),
          ("templated", slice(495, 514, None), slice(108, 109, None)),
          ("literal", slice(514, 515, None), slice(109, 110, None)),
          ("templated", slice(515, 531, None), slice(110, 112, None)),
          ("block_start", slice(531, 565, None), slice(112, 112, None)),
          ("literal", slice(565, 570, None), slice(112, 117, None)),
          ("block_end", slice(570, 578, None), slice(117, 117, None)),
          ("literal", slice(578, 579, None), slice(117, 118, None)),
          ("block_end", slice(579, 592, None), slice(118, 118, None)),
          ("literal", slice(469, 477, None), slice(118, 126, None)),
          ("templated", slice(477, 494, None), slice(126, 136, None)),
          ("literal", slice(494, 495, None), slice(136, 137, None)),
          ("templated", slice(495, 514, None), slice(137, 138, None)),
          ("literal", slice(514, 515, None), slice(138, 139, None)),
          ("templated", slice(515, 531, None), slice(139, 144, None)),
          ("block_start", slice(531, 565, None), slice(144, 144, None)),
          ("block_end", slice(570, 578, None), slice(144, 144, None)),
          ("literal", slice(578, 579, None), slice(144, 145, None)),
          ("block_end", slice(579, 592, None), slice(145, 145, None)),
          ("literal", slice(592, 594, None), slice(145, 147, None))],
         CheetahTemplater),
        ("""
## 行内end if（cheetah会吸收000）
#set $columns = ["id", "name", "age", "department"]
#for $index, $column in enumerate($columns)
    $column#if $index < len($columns) - 1#, #else if $index == len($columns) - 1# 最后 #else# 其他 #end if00
#end for00

""",
         None,
         [("literal", slice(0, 1, None), slice(0, 1, None)), ("templated", slice(1, 28, None), slice(1, 1, None)),
          ("templated", slice(28, 80, None), slice(1, 1, None)),
          ("block_start", slice(80, 124, None), slice(1, 1, None)),
          ("literal", slice(124, 128, None), slice(1, 5, None)),
          ("templated", slice(128, 135, None), slice(5, 7, None)),
          ("block_start", slice(135, 166, None), slice(7, 7, None)),
          ("literal", slice(166, 168, None), slice(7, 9, None)),
          ("block_mid", slice(168, 205, None), slice(9, 9, None)),
          ("block_mid", slice(209, 215, None), slice(9, 9, None)),
          ("block_end", slice(219, 228, None), slice(9, 9, None)),
          ("literal", slice(228, 229, None), slice(9, 10, None)),
          ("block_end", slice(229, 240, None), slice(10, 10, None)),
          ("literal", slice(124, 128, None), slice(10, 14, None)),
          ("templated", slice(128, 135, None), slice(14, 18, None)),
          ("block_start", slice(135, 166, None), slice(18, 18, None)),
          ("literal", slice(166, 168, None), slice(18, 20, None)),
          ("block_mid", slice(168, 205, None), slice(20, 20, None)),
          ("block_mid", slice(209, 215, None), slice(20, 20, None)),
          ("block_end", slice(219, 228, None), slice(20, 20, None)),
          ("literal", slice(228, 229, None), slice(20, 21, None)),
          ("block_end", slice(229, 240, None), slice(21, 21, None)),
          ("literal", slice(124, 128, None), slice(21, 25, None)),
          ("templated", slice(128, 135, None), slice(25, 28, None)),
          ("block_start", slice(135, 166, None), slice(28, 28, None)),
          ("literal", slice(166, 168, None), slice(28, 30, None)),
          ("block_mid", slice(168, 205, None), slice(30, 30, None)),
          ("block_mid", slice(209, 215, None), slice(30, 30, None)),
          ("block_end", slice(219, 228, None), slice(30, 30, None)),
          ("literal", slice(228, 229, None), slice(30, 31, None)),
          ("block_end", slice(229, 240, None), slice(31, 31, None)),
          ("literal", slice(124, 128, None), slice(31, 35, None)),
          ("templated", slice(128, 135, None), slice(35, 45, None)),
          ("block_start", slice(135, 166, None), slice(45, 45, None)),
          ("block_mid", slice(168, 205, None), slice(45, 45, None)),
          ("literal", slice(205, 209, None), slice(45, 49, None)),
          ("block_mid", slice(209, 215, None), slice(49, 49, None)),
          ("block_end", slice(219, 228, None), slice(49, 49, None)),
          ("literal", slice(228, 229, None), slice(49, 50, None)),
          ("block_end", slice(229, 240, None), slice(50, 50, None)),
          ("literal", slice(240, 241, None), slice(50, 51, None))],
         CheetahTemplater),
        ("""
    #def date_filter($delta=0, $extra_delta=0, $reload_delta=0, $dt_key='dt')
#if 1 == 1
    $dt_key = '2024'
  #end if
#end def

    select
      algo_versio
      from mart_peisongturing.fact_banma_algorithm_tracing_hour
      where ${date_filter(0, 0, 1)}""",
         None,
         [("literal", slice(0, 1, None), slice(0, 1, None)), ("block_start", slice(1, 79, None), slice(1, 1, None)),
          ("block_start", slice(79, 90, None), slice(1, 1, None)), ("literal", slice(90, 94, None), slice(1, 1, None)),
          ("templated", slice(94, 101, None), slice(1, 1, None)), ("literal", slice(101, 111, None), slice(1, 1, None)),
          ("block_end", slice(111, 121, None), slice(1, 1, None)),
          ("block_end", slice(121, 130, None), slice(1, 1, None)),
          ("literal", slice(130, 236, None), slice(1, 107, None)),
          ("templated", slice(236, 259, None), slice(107, 123, None))],
         CheetahTemplater),
        ("""
    #def date_filter($delta=0, $extra_delta=0, $reload_delta=0, $dt_key='dt')
      #def inner_filter()
        #if 1 == 1
          $dt_key = '2024'
        #end if
      #end def
      ${inner_filter()}
    #end def

    select
      algo_versio
    from mart_peisongturing.fact_banma_algorithm_tracing_hour
    where ${date_filter(0, 0, 1)}
    """,
         None,
         [("literal", slice(0, 1, None), slice(0, 1, None)), ("block_start", slice(1, 79, None), slice(1, 1, None)),
          ("block_start", slice(79, 105, None), slice(1, 1, None)),
          ("block_start", slice(105, 124, None), slice(1, 1, None)),
          ("literal", slice(124, 134, None), slice(1, 1, None)),
          ("templated", slice(134, 141, None), slice(1, 1, None)),
          ("literal", slice(141, 151, None), slice(1, 1, None)),
          ("block_end", slice(151, 167, None), slice(1, 1, None)),
          ("block_end", slice(167, 182, None), slice(1, 1, None)),
          ("literal", slice(182, 188, None), slice(1, 1, None)),
          ("templated", slice(188, 205, None), slice(1, 1, None)),
          ("literal", slice(205, 206, None), slice(1, 1, None)),
          ("block_end", slice(206, 219, None), slice(1, 1, None)),
          ("literal", slice(219, 321, None), slice(1, 103, None)),
          ("templated", slice(321, 344, None), slice(103, 132, None)),
          ("literal", slice(344, 349, None), slice(132, 137, None))],
         CheetahTemplater),
        ("""
#set $i = 0
#set $max_partitions = 5
#while $i < $max_partitions
    SELECT * FROM my_table
    WHERE partition_key = $i;
    #set $i = $i + 1
#end while
""",
         None,
         [("literal", slice(0, 1, None), slice(0, 1, None)), ("templated", slice(1, 13, None), slice(1, 1, None)),
          ("templated", slice(13, 38, None), slice(1, 1, None)),
          ("block_start", slice(38, 66, None), slice(1, 1, None)),
          ("literal", slice(66, 119, None), slice(1, 54, None)),
          ("templated", slice(119, 121, None), slice(54, 55, None)),
          ("literal", slice(121, 123, None), slice(55, 57, None)),
          ("templated", slice(123, 144, None), slice(57, 57, None)),
          ("block_end", slice(144, 155, None), slice(57, 57, None)),
          ("literal", slice(66, 119, None), slice(57, 110, None)),
          ("templated", slice(119, 121, None), slice(110, 111, None)),
          ("literal", slice(121, 123, None), slice(111, 113, None)),
          ("templated", slice(123, 144, None), slice(113, 113, None)),
          ("block_end", slice(144, 155, None), slice(113, 113, None)),
          ("literal", slice(66, 119, None), slice(113, 166, None)),
          ("templated", slice(119, 121, None), slice(166, 167, None)),
          ("literal", slice(121, 123, None), slice(167, 169, None)),
          ("templated", slice(123, 144, None), slice(169, 169, None)),
          ("block_end", slice(144, 155, None), slice(169, 169, None)),
          ("literal", slice(66, 119, None), slice(169, 222, None)),
          ("templated", slice(119, 121, None), slice(222, 223, None)),
          ("literal", slice(121, 123, None), slice(223, 225, None)),
          ("templated", slice(123, 144, None), slice(225, 225, None)),
          ("block_end", slice(144, 155, None), slice(225, 225, None)),
          ("literal", slice(66, 119, None), slice(225, 278, None)),
          ("templated", slice(119, 121, None), slice(278, 279, None)),
          ("literal", slice(121, 123, None), slice(279, 281, None)),
          ("templated", slice(123, 144, None), slice(281, 281, None)),
          ("block_end", slice(144, 155, None), slice(281, 281, None))],
         CheetahTemplater)
    ],
    ids=["empty", "literal", "placeholder", "loop", "conditional", "if in for", "inline end block", "def",
         "nested def", "while"],
)
def test__templater_cheetah_slice_file(
    raw_file, override_context, result, templater_class, caplog
):
    """Test slice_file."""
    templater = templater_class(override_context=override_context)
    render_func = templater.construct_render_func()

    with caplog.at_level(logging.DEBUG, logger="sqlfluff.templater"):
        raw_sliced, sliced_file, templated_str = templater.slice_file(
            raw_file, render_func=render_func
        )
    # Create a TemplatedFile from the results. This runs some useful sanity
    # checks.
    _ = TemplatedFile(raw_file, "<<DUMMY>>", templated_str, sliced_file, raw_sliced)
    # Check contiguous on the TEMPLATED VERSION
    # print(sliced_file)
    prev_slice = None
    for elem in sliced_file:
        # print(elem)
        if prev_slice:
            assert elem[2].start == prev_slice.stop
        prev_slice = elem[2]
    # Check that all literal segments have a raw slice
    for elem in sliced_file:
        if elem[0] == "literal":
            assert elem[1] is not None
    # check result
    actual = [
        (
            templated_file_slice.slice_type,
            templated_file_slice.source_slice,
            templated_file_slice.templated_slice,
        )
        for templated_file_slice in sliced_file
    ]
    assert actual == result
