import subprocess
from ribohmm import contrib as bam_to_tbi


class TestBamToTbi(object):
    def test_which(self):
        assert bam_to_tbi.which('bash') == '/bin/bash'
        assert bam_to_tbi.which('gcc') == '/usr/bin/gcc'

    def test_convert_rnaseq(self):
        pass

    def test_convert_riboseq(self):
        bam_to_tbi.convert_riboseq(
            bam_file='test_data/riboseq.test.bam',
            bgzip_path=bam_to_tbi.which('bgzip'),
            tabix_path=bam_to_tbi.which('tabix')
        )

        generated_files = [
            'riboseq.test_{}.tbx.{}.{}'.format(direction, n, ext)
            for direction in ('fwd', 'rev')
            for n in (28, 29, 30, 31)
            for ext in ('gz', 'gz.tbi')
        ]

        for gen_file in generated_files:
            assert subprocess.call(
                'diff test_data/{} test_data/truth/{}'.format(
                    gen_file,
                    gen_file.replace('.tbx', '')
                ),
                shell=True
            ) == 0
