from video_processor import dbAllocator
import tqdm

al = dbAllocator.Allocator('sqlite:///../testv-complete.sqlite', '/')
per = al.queryAllPerson()
for p in tqdm.tqdm(per):
	fa = al.queryFaceByPerson(p)
	if len(fa) == 0:
		p.keyface = None
		continue
	maxFace = max(fa, key=lambda f: (f.x2 - f.x1) * (f.y2 - f.y1))
	p.keyface = maxFace

al.session.commit()