from pydub import AudioSegment as aseg
from pydub.utils import make_chunks
import os
import glob

chunk_length = 5000
indir = 'Data\Test'
outdir = 'DataSplit_test\Test'

file_ext = '*.wav'
sub_dirs = os.listdir(indir)

sub_dirs.sort()

print(sub_dirs)

for label,sub_dir in enumerate(sub_dirs):
    if(os.path.isdir(os.path.join(indir, sub_dir))):
        for file_name in glob.glob(os.path.join(indir, sub_dir, file_ext)):
            try:
                if file_name:
                    print('Splitting file: ', file_name)
                    myaudio = aseg.from_file(file_name, "wav")
                    chunks = make_chunks(myaudio, chunk_length)

                    out_file_name = file_name.split('\\')[3]

                    out_subdir = outdir + '\\' + sub_dir + '\\'

                    if not os.path.exists(out_subdir):
                        os.makedirs(out_subdir)

                    out_file = out_subdir + out_file_name
                    out_file = out_file.replace('.wav', '')

                    for i, chunk in enumerate(chunks):
                        chunk_name = out_file + '{0}.wav'
                        chunk_name = chunk_name.format(i)
                        print('Exporting',chunk_name)
                        chunk.export(chunk_name, format= "wav")

            except Exception as e:
                print("[Error] Splitting error in %s. %s" % (file_name, e))
                continue

